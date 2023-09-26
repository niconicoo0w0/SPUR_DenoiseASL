import argparse
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM3D, BatchNormalization, Activation, Permute, Conv3D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

def get_nifti_data(path):
    nifti_file = nib.load(path)
    data = nifti_file.get_fdata()
    # Move the timepoints to the first dimension
    # data.ndim == 4: (91, 109, 91, 168) => (168, 91, 109, 91, 1)
    if data.ndim == 4:
        data = np.expand_dims(data, axis = -2)
        data = np.transpose(data, (4, 0, 1, 2, 3))
        
    # data.ndim == 3: (91, 109, 91) => (1, 91, 109, 91, 1)
    if data.ndim == 3:
        data = np.expand_dims(np.expand_dims(data, axis = 0), axis = -1)

    # mean = np.mean(data)
    # std = np.std(data)
    # return (data - mean) / std
    mean = np.mean(data, axis=(1, 2, 3, 4), keepdims=True)
    std = np.std(data, axis=(1, 2, 3, 4), keepdims=True)
    small_const = 1e-7
    return (data - mean) / (std + small_const)

def convlstm_block(input_layer, num_filters):
    convlstm = ConvLSTM3D(num_filters, (3, 3, 3), padding='same', return_sequences=True)(input_layer)
    norm = BatchNormalization()(convlstm)
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
    
    print("get nifti data and reshape")
    
    input_data = [get_nifti_data(path) for path in input_paths]
    input_data = np.stack(input_data, axis=0) 
    target_data = [get_nifti_data(path) for path in target_paths]
    target_data = np.stack(target_data, axis=0)
    
    print("start split")

    X_data, X_test, y_data, y_test = train_test_split(input_data, target_data, test_size=1 - train_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_ratio/(test_ratio + validation_ratio)) 
    
    print("layer")
    
    num_subj = np.shape(input_data)[0]
    input_shape = (num_subj, 168, 91, 109, 91, 1)   #(6, 168, 91, 109, 91, 1)
    input_layer = Input(input_shape[1:])            #(None, 168, 91, 109, 91, 1)
    block1 = convlstm_block(input_layer, 8)
    block2 = convlstm_block(block1, 16)
    final = ConvLSTM3D(1, (3, 3, 3), padding='same', return_sequences=True)(block2)
    
    print("training model")
    
    training_model = Model(inputs=[input_layer], outputs=[final])
    training_model.summary()

    training_model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    
    print("fit")
    
    training_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=1, callbacks=[early_stopping])
    
    for i, path in enumerate(input_paths):
        input_data = get_nifti_data(path)
        input_data = np.expand_dims(input_data, axis=0)
        denoised_image = training_model.predict(input_data)
        # (num_subj, 168, 91, 109, 91, 1) => (num_subj, 91, 109, 91, 1, 168)
        print(np.shape(denoised_image))
        denoised_image = np.transpose(denoised_image, (0, 2, 3, 4, 5, 1))
        denoised_image = denoised_image.squeeze()
        save_as_nifti(denoised_image, os.path.join(args.output, f'denoised_{i}.nii.gz'), path)

if __name__ == "__main__":
    main()
