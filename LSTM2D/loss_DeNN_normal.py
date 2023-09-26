import argparse
import argparse
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization

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
        data = np.transpose(data, (3, 1, 0, 2, 4))
        
    mean = np.mean(data, axis=(1), keepdims=True)
    std = np.std(data, axis=(1), keepdims=True)
    return (data - mean) / (std + 1e-6)

def convlstm_block(input_layer, num_filters):
    convlstm = ConvLSTM2D(num_filters, (3, 3), padding='same', return_sequences=True)(input_layer)
    norm = LayerNormalization()(convlstm)
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

# custom loss function
def denoise_loss_ASL(y_true, y_pred):
    denoised_ASL = y_pred[:,:,:,0,:]
    original_ASL = y_true[:,:,:,0,:]

    tdim = tf.shape(denoised_ASL)[1]

    denoised_ASL = denoised_ASL - tf.reduce_mean(denoised_ASL, axis=-1, keepdims=True)
    original_ASL = original_ASL - tf.reduce_mean(original_ASL, axis=-1, keepdims=True)

    denoised_ASL = denoised_ASL / (tf.math.reduce_std(denoised_ASL, axis=-1, keepdims=True) + 1e-6)
    original_ASL = original_ASL / (tf.math.reduce_std(original_ASL, axis=-1, keepdims=True) + 1e-6) 

    corr_mat = tf.reduce_sum(tf.multiply(denoised_ASL, original_ASL), axis=1) / (tf.cast(tdim, tf.float32) + 1e-6)
    loss = tf.reduce_mean(tf.abs(corr_mat))
    return loss

def custom_loss(training_model):
    def loss(y_true, y_pred):
        alpha_ = 1.0
        beta_ = 0.1
        gamma_ = 1.0
        lambda_ = 0.01

        reconstruction_loss = K.mean(K.square(y_true - y_pred))
        y_pred_right_shifted = K.concatenate([y_pred[:, 1:, :], y_pred[:, -1:, :]], axis=1)
        temporal_consistency_loss = K.mean(K.square(y_pred - y_pred_right_shifted))
        denoise_ASL_loss = denoise_loss_ASL(y_true, y_pred)
        regularization_term = lambda_ * K.sum([K.sum(K.square(w)) for w in training_model.trainable_weights])
        return alpha_ * reconstruction_loss + beta_ * temporal_consistency_loss + gamma_ * denoise_ASL_loss + regularization_term
    return loss

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
    
    block1 = convlstm_block(input_layer, 32)
    # block2 = convlstm_block(block1, 32)
    # block3 = convlstm_block(block2, 64)
    final = ConvLSTM2D(1, (3, 3), padding='same', return_sequences=True)(block1)

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

    training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=custom_loss(training_model=training_model))  # Reduced learning rate
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    training_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32, callbacks=[early_stopping])

    for i, path in enumerate(input_paths):
        input_data = get_nifti_data(path)
        denoised_image = predict_in_batches(training_model, input_data, num_subj, 91*num_subj)
        # (91, 168, 91, 109, 1) => (91, 168, 91, 109)
        denoised_image = np.squeeze(denoised_image)
        # transpose => (91, 109, 91, 168)
        denoised_image = np.transpose(denoised_image, (2, 3, 0, 1))
        save_as_nifti(denoised_image, os.path.join(args.output, f'denoised_{i}.nii.gz'), path)

if __name__ == "__main__":
    main()
