import argparse
import os
from tensorflow.keras.models import Model
from keras.layers import Input, Conv3D, BatchNormalization, Activation, concatenate
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

def get_nifti_data(path):
    nifti_file = nib.load(path)
    return nifti_file.get_fdata()

def get_mean(path):
    data = get_nifti_data(path)
    return np.mean(data, axis=3)

def get_variance(path):
    data = get_nifti_data(path)
    return np.var(data, axis=3)

def reshape_input(data):
    data = np.expand_dims(data, axis=-1)
    data = np.expand_dims(data, axis=0)
    return data

def conv_block(input_layer, num_filters):
    conv = Conv3D(num_filters, (3, 3, 3), padding='same')(input_layer)
    norm = BatchNormalization()(conv)
    relu = Activation('relu')(norm)
    conv = Conv3D(num_filters, (3, 3, 3), padding='same')(relu)
    norm = BatchNormalization()(conv)
    relu = Activation('relu')(norm)
    conv = Conv3D(num_filters, (3, 3, 3), padding='same')(relu)
    return BatchNormalization()(conv)

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
    validation_ratio = 0.15
    test_ratio = 0.15

    input_paths = args.data
    target_paths = args.target

    # Concatenate the data
    input_data = [get_mean(path) for path in input_paths]
    input_data = [reshape_input(data) for data in input_data]
    target_data = [get_nifti_data(path) for path in target_paths]
    target_data = [reshape_input(data) for data in target_data]

    # Split the data
    X_data, X_test, y_data, y_test = train_test_split(input_data, target_data, test_size=1 - train_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_ratio/(test_ratio + validation_ratio))

    input_shape = np.shape(X_train[0])[1:]

    input_layer = Input(input_shape)

    block = conv_block(input_layer, 64)
    final = Conv3D(1, (3, 3, 3), padding='same')(block)

    model = Model(inputs=[input_layer], outputs=[final])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16)
    denoised_data = model.predict(X_test)
    
    for i in range(len(denoised_data)):
        denoised_image = denoised_data[i].squeeze()
        save_as_nifti(denoised_image, args.output + f'/denoised_{i}.nii', input_paths[i])

if __name__ == "__main__":
    main()
