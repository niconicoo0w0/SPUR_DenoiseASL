import argparse
import os
from tensorflow.keras.models import Model
from keras.layers import Input, Conv3D, BatchNormalization, Activation
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
    
def get_nifti_data(path):
    nifti_file = nib.load(path)
    data = nifti_file.get_fdata()

    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def reshape_input(data):
    # fix (109, 91, 168) => (91, 109, 91, 168)
    if data.ndim == 4:
        return np.expand_dims(data, axis=0)
    elif data.ndim == 3:
        return np.expand_dims(np.expand_dims(data, axis=-1), axis=0)
    else:
        raise ValueError(f"Invalid number of dimensions {data.ndim}. The data should be 3D or 4D.")

def conv_block(input_layer, num_filters):
    conv = Conv3D(num_filters, (3, 3, 3), padding='same')(input_layer)
    norm = BatchNormalization()(conv)
    relu = Activation('relu')(norm)
    return relu

def save_as_nifti(image, output_path, original_nifti_path):
    original_nifti = nib.load(original_nifti_path)
    denoised_nifti = nib.Nifti1Image(image, original_nifti.affine, original_nifti.header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(denoised_nifti, output_path)

def main():
    parser = argparse.ArgumentParser(description='Train CNN with ASL data.')
    parser.add_argument('--data', nargs='+', type=str, help='List of paths to the ASL data.')
    parser.add_argument('--target', nargs='+', type=str, help='List of paths to the target (gold standard) data.')
    parser.add_argument('--output', type=str, help='Path to save the denoised data.')
    args = parser.parse_args()

    train_ratio = 0.7
    validation_ratio = 0.2
    test_ratio = 0.1

    input_paths = args.data
    target_paths = args.target
    
    # fix more than 1 input
    input_data = [get_nifti_data(path) for path in input_paths]
    input_data = np.concatenate([reshape_input(data) for data in input_data], axis=0)
    target_data = [get_nifti_data(path) for path in target_paths]
    target_data = np.concatenate([reshape_input(data) for data in target_data], axis=0)
    
    print("input_data", np.shape(input_data))
    print("target_data", np.shape(target_data))
    
    # Split the data
    X_data, X_test, y_data, y_test = train_test_split(input_data, target_data, test_size=1 - train_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_ratio/(test_ratio + validation_ratio))
    
    num_subj = np.shape(X_train)[0]
    input_shape = (num_subj, 91, 109, 91, 168)  # fix (num_subj, 109, 91, 168) => (num_subj, 91, 109, 91, 168)
    input_layer = Input(input_shape[1:])
    block1 = conv_block(input_layer, 32)
    block2 = conv_block(block1, 64)
    final = Conv3D(168, (1, 1, 1), padding='same')(block2)
    
    # Train
    training_model = Model(inputs=[input_layer], outputs=[final])
    training_model.summary()
 
    training_model.compile(optimizer='adam', loss='mean_squared_error')
    training_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=2)
    
    for i, path in enumerate(input_paths):
        input_data = reshape_input(get_nifti_data(path))
        denoised_image = training_model.predict(input_data)
        denoised_image = denoised_image.squeeze()
        save_as_nifti(denoised_image, os.path.join(args.output, f'denoised_{i}.nii.gz'), path)

if __name__ == "__main__":
    main()


# from sklearn.model_selection import GroupShuffleSplit

# # Assume groups is a list or array of same length as input_data and target_data,
# # indicating the group (e.g., patient ID) of each data point
# gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, test_size=1-train_ratio)
# train_indices, test_indices = next(gss.split(input_data, target_data, groups=groups))

# X_train, y_train = input_data[train_indices], target_data[train_indices]
# X_test, y_test = input_data[test_indices], target_data[test_indices]

# # Now split the non-test data into training and validation sets
# gss_val = GroupShuffleSplit(n_splits=1, train_size=(1-test_ratio)/(1-test_ratio+validation_ratio), test_size=validation_ratio/(1-test_ratio+validation_ratio))
# train_indices, val_indices = next(gss_val.split(X_train, y_train, groups=train_indices))

# X_train, y_train = X_train[train_indices], y_train[train_indices]
# X_val, y_val = X_train[val_indices], y_train[val_indices]
