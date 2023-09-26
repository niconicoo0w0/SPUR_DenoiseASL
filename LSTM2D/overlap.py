import argparse
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Activation, Permute, Conv2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

index_to_split = 2
chunck_size = 13
total_size = 91
num_timepoint = 168
overlap = 5  # Overlap size

exp_shape = (num_timepoint, chunck_size, 109, 1)

def get_nifti_data(path, chunk_size=chunck_size, overlap=5):
    nifti_file = nib.load(path)
    data = nifti_file.get_fdata()
    limit = total_size
    if data.ndim == 4:
        mean = np.mean(data, axis=(0, 1, 2, 3), keepdims=True)
        std = np.std(data, axis=(0, 1, 2, 3), keepdims=True)
        data = (data - mean) / (std + 1e-7)
        data = np.transpose(data, (2, 3, 0, 1))
    elif data.ndim == 3:
        mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
        std = np.std(data, axis=(0, 1, 2), keepdims=True)
        data = (data - mean) / (std + 1e-7)
        data = np.expand_dims(data, axis=1)
        data = np.repeat(data, num_timepoint, axis=1)
        data = np.transpose(data, (3, 1, 0, 2))
    for i in range(0, limit, chunk_size - overlap):
        chunk = data[:, :, i:min(i+chunk_size, limit), :]
        if chunk.shape[index_to_split] < chunk_size:
            padding_shape = list(chunk.shape)
            padding_shape[index_to_split] = chunk_size - chunk.shape[index_to_split]
            padding = np.zeros(padding_shape)
            chunk = np.concatenate([chunk, padding], axis=index_to_split)
        chunk = np.expand_dims(chunk, axis=-1)
        print(np.shape(chunk))
        yield chunk

def reassemble_image(image_chunks, overlap):
    num_chunks = len(image_chunks)
    chunk_size = image_chunks[0].shape[2]
    total_size = num_chunks * (chunk_size - overlap) + overlap
    reassembled_image = np.zeros((91, num_timepoint, total_size, 109, 1)) # added a dimension here
    counts = np.zeros((91, num_timepoint, total_size, 109, 1))  # added a dimension here
    for i, chunk in enumerate(image_chunks):
        start = i * (chunk_size - overlap)
        end = start + chunk_size
        reassembled_image[:, :, start:end, :, :] += chunk  # adjusted slicing here
        counts[:, :, start:end, :, :] += 1  # adjusted slicing here
    return reassembled_image / counts



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
    
    input_data = [chunk for path in input_paths for chunk in get_nifti_data(path)]
    input_data = np.vstack(input_data)
    input_data = np.reshape(input_data, (-1, num_timepoint, chunck_size, 109, 1))
    print("input", np.shape(input_data))
    
    target_data = [chunk for path in target_paths for chunk in get_nifti_data(path)]
    target_data = np.vstack(target_data)
    target_data = np.reshape(target_data, (-1, num_timepoint, chunck_size, 109, 1))
    print("target", np.shape(target_data))

    X_data, X_test, y_data, y_test = train_test_split(input_data, target_data, test_size= 1 - train_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=False)
    
    input_shape = exp_shape
    
    # (None, 168, 91, 109, 1)
    input_layer = Input(input_shape)
    
    block1 = convlstm_block(input_layer, 16)
    block2 = convlstm_block(block1, 32)
    final = ConvLSTM2D(1, (3, 3), padding='same', return_sequences=True)(block2)

    print("X_data",np.shape(X_data))
    print("y_data",np.shape(y_data))
    print("X_train",np.shape(X_train))
    print("X_val",np.shape(X_val))
    print("y_train",np.shape(y_train))
    print("y_val",np.shape(y_val))

    training_model = Model(inputs=[input_layer], outputs=[final])
    training_model.summary()

    training_model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    training_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64, callbacks=[early_stopping])
    
    num_subj = np.shape(input_data)[0]
    for i, path in enumerate(input_paths):
        denoised_image_chunks = []
        for input_chunk in get_nifti_data(path):
            denoised_chunk = predict_in_batches(training_model, input_chunk, num_subj, 91*num_subj)
            denoised_image_chunks.append(denoised_chunk)
        denoised_image = reassemble_image(denoised_image_chunks, overlap)
        denoised_image = np.squeeze(denoised_image)
        denoised_image = np.transpose(denoised_image, (2, 3, 0, 1))
        save_as_nifti(denoised_image, os.path.join(args.output, f'denoised_{i}.nii.gz'), path)

if __name__ == "__main__":
    main()
