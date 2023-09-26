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

def target_estimator(normalized_data):
    means = np.mean(normalized_data, axis=(1, 2, 3))
    stds = np.std(normalized_data, axis=(1, 2, 3))

    estimated_target = np.empty_like(normalized_data)
    for t in range(normalized_data.shape[0]):
        estimated_target[t] = (normalized_data[t] - means[t]) / stds[t]

    return estimated_target

def get_nifti_data_and_target(path, target_estimator):
    nifti_file = nib.load(path)
    data = nifti_file.get_fdata()
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    estimated_target = target_estimator(normalized_data)
    return normalized_data, estimated_target

def reshape_input(data):
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
    parser.add_argument('--output', type=str, help='Path to save the denoised data.')
    args = parser.parse_args()

    train_ratio = 0.7
    validation_ratio = 0.2
    test_ratio = 0.1

    input_paths = args.data
    
    input_data, target_data = [], []
    for path in input_paths:
        data, estimated_target = get_nifti_data_and_target(path, target_estimator)
        input_data.append(reshape_input(data))
        target_data.append(reshape_input(estimated_target))
    input_data = np.concatenate(input_data, axis=0)
    target_data = np.concatenate(target_data, axis=0)

    print("input_data", np.shape(input_data))
    print("target_data", np.shape(target_data))

    X_data, X_test, y_data, y_test = train_test_split(input_data, target_data, test_size=1 - train_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_ratio/(test_ratio + validation_ratio))
    
    num_subj = np.shape(X_train)[0]
    input_shape = (num_subj, 91, 109, 91, 168)
    input_layer = Input(input_shape[1:])
    block1 = conv_block(input_layer, 32)
    block2 = conv_block(block1, 64)
    final = Conv3D(168, (1, 1, 1), padding='same', activation='relu')(block2)
    
    training_model = Model(inputs=[input_layer], outputs=[final])
    training_model.summary()
 
    training_model.compile(optimizer='adam', loss='mean_squared_error')
    training_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=2)
    
    for i, path in enumerate(input_paths):
        input_data, _ = get_nifti_data_and_target(path, target_estimator)
        input_data = reshape_input(input_data)
        denoised_image = training_model.predict(input_data)
        denoised_image = denoised_image.squeeze()
        save_as_nifti(denoised_image, os.path.join(args.output, f'denoised_{i}.nii.gz'), path)

if __name__ == "__main__":
    main()
