import argparse
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Activation, Permute, Conv2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

def get_nifti_data(path):
    nifti_file = nib.load(path)
    data = nifti_file.get_fdata()
    if data.ndim == 4:
        # (91, 109, 91, 168) => (91, 168, 91, 109)
        data = np.transpose(data, (2, 3, 0, 1))
        # (91, 168, 91, 109) => (91, 168, 91, 109, 1)
        data = np.expand_dims(data, axis=-1)
    if data.ndim == 3:
        # (91, 109, 91) => (91, 1, 109, 91, 1)
        data = np.expand_dims(np.expand_dims(data, axis = 1), axis = -1)
        # (91, 1, 109, 91, 1) => (91, 1, 91, 109, 1)
        data = np.transpose(data, (0, 1, 3, 2, 4))
        
    mean = np.mean(data, axis=(1, 2, 3), keepdims=True)
    std = np.std(data, axis=(1, 2, 3), keepdims=True)
    small_const = 1e-7
    return (data - mean) / (std + small_const)

def convlstm_block(input_layer, num_filters):
    convlstm = ConvLSTM2D(num_filters, (3, 3), padding='same', return_sequences=True)(input_layer)
    norm = BatchNormalization()(convlstm)
    relu = Activation('relu')(norm)
    return relu

def save_as_nifti(image, output_path, original_nifti_path):
    original_nifti = nib.load(original_nifti_path)
    denoised_nifti = nib.Nifti1Image(image, original_nifti.affine, original_nifti.header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(denoised_nifti, output_path)

def predict_in_batches(model, input_data, num_subj, batch_size):
    predictions = []
    for i in range(0, num_subj, batch_size):
        batch = input_data[i : min(i + batch_size, num_subj)]
        batch_predictions = model.predict(batch)
        predictions.append(batch_predictions)
    return np.concatenate(predictions, axis=0)

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

    input_data = [get_nifti_data(path) for path in input_paths]
    input_data = np.vstack(input_data)
    target_data = [get_nifti_data(path) for path in target_paths]
    target_data = np.vstack(target_data)
    
    print("input", np.shape(input_data))
    print("target", np.shape(target_data))
    # input (546, 168, 91, 109, 1)
    # target (546, 1, 91, 109, 1)

    X_data, X_test, y_data, y_test = train_test_split(input_data, target_data, test_size= 1 - train_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=False)
    
    num_subj = np.shape(input_data)[0]
    # data shape would be (91*num_subjects,168,91,109,1)
    input_shape = (91*num_subj, 168, 91, 109, 1)
    
    # (None, 91*num_subjects, 168, 91, 109, 1)
    input_layer = Input(input_shape[1:])
    
    block1 = convlstm_block(input_layer, 16)
    block2 = convlstm_block(block1, 32)
    # block3 = convlstm_block(block2, 64)
    final = ConvLSTM2D(1, (3, 3), padding='same', return_sequences=True)(block2)

    # X_data (382, 168, 91, 109, 1)
    # y_data (382, 1, 91, 109, 1)
    # X_train (254, 168, 91, 109, 1)
    # X_val (128, 168, 91, 109, 1)
    # y_train (254, 1, 91, 109, 1)
    # y_val (128, 1, 91, 109, 1)

    print("X_data",np.shape(X_data))
    print("y_data",np.shape(y_data))
    print("X_train",np.shape(X_train))
    print("X_val",np.shape(X_val))
    print("y_train",np.shape(y_train))
    print("y_val",np.shape(y_val))

    training_model = Model(inputs=[input_layer], outputs=[final])
    training_model.summary()

    training_model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    training_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=16, callbacks=[early_stopping])

    for i, path in enumerate(input_paths):
        input_data = get_nifti_data(path)
        print("ok109")
        denoised_image = predict_in_batches(training_model, input_data, num_subj, 91*num_subj)
        print("denoised_image", np.shape(denoised_image))
        # (91, 168, 91, 109, 1) => (91, 168, 91, 109)
        denoised_image = np.squeeze(denoised_image)
        # transpose => (91, 109, 91, 168)
        denoised_image = np.transpose(denoised_image, (2, 3, 0, 1))
        print("denoised_image after", np.shape(denoised_image))
        save_as_nifti(denoised_image, os.path.join(args.output, f'denoised_{i}.nii.gz'), path)

if __name__ == "__main__":
    main()
