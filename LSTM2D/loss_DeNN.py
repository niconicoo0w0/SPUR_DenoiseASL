import argparse
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Activation, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from datetime import datetime
import nibabel as nib
import scipy as scp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import keras.backend as T
import tensorflow_probability as tfp


SUBIDS=['MCI1007','MCI1008','MCI1017','MCI1019_120121','MCI1020_112321','MCI1037_042122']

index_to_split = 2
chunk_size = 91
total_size = 91
length = 109
num_timepoint = 168
overlap = 0  # Overlap size

exp_shape = (num_timepoint, chunk_size, length, 1)

def get_nifti_data(subj, chunk_size=chunk_size, overlap=0, target=0):
    
    if target == 1:
        nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss_mean.nii.gz')
    else:
        nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss.nii.gz')
        data = scp.stats.zscore(nifti_file.get_fdata(),axis=3)

    #nifti_file = nib.load(path)
    data = nifti_file.get_fdata()
    limit = total_size
    if data.ndim == 4:
        mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
        std = np.std(data, axis=(0, 1, 2), keepdims=True)
        data = (data - mean) / (std + 1e-6)
        data = np.transpose(data, (2, 3, 0, 1))
    elif data.ndim == 3:
        mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
        std = np.std(data, axis=(0, 1, 2), keepdims=True)
        data = (data - mean) / (std + 1e-6)
        data = np.expand_dims(data, axis=1)
        data = np.transpose(data, (3, 1, 0, 2))
    for i in range(0, limit, chunk_size - overlap):
        chunk = data[:, :, i:min(i+chunk_size, limit), :]
        if chunk.shape[index_to_split] < chunk_size:
            padding_shape = list(chunk.shape)
            padding_shape[index_to_split] = chunk_size - chunk.shape[index_to_split]
            padding = np.zeros(padding_shape)
            chunk = np.concatenate([chunk, padding], axis=index_to_split)
        chunk = np.expand_dims(chunk, axis=-1)
        #print(np.shape(chunk))
        return chunk

def reassemble_image(image_chunks, overlap):
    num_chunks = len(image_chunks)
    chunk_size = image_chunks[0].shape[2]
    total_size = num_chunks * (chunk_size - overlap) + overlap
    reassembled_image = np.zeros((chunk_size, num_timepoint, total_size, length, 1)) # added a dimension here
    counts = np.zeros((chunk_size, num_timepoint, total_size, length, 1))  # added a dimension here
    for i, chunk in enumerate(image_chunks):
        start = i * (chunk_size - overlap)
        end = start + chunk_size
        reassembled_image[:, :, start:end, :, :] += chunk  # adjusted slicing here
        counts[:, :, start:end, :, :] += 1  # adjusted slicing here
    return reassembled_image / counts

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
        gamma_ = 1.5
        lambda_ = 0.05
        
        reconstruction_loss = K.mean(K.square(y_true - y_pred))
        y_pred_right_shifted = K.concatenate([y_pred[:, 1:, :], y_pred[:, -1:, :]], axis=1)
        temporal_consistency_loss = K.mean(K.square(y_pred - y_pred_right_shifted))
        denoise_ASL_loss = denoise_loss_ASL(y_true, y_pred)
        regularization_term = lambda_ * K.sum([K.sum(K.square(w)) for w in training_model.trainable_weights])
        return alpha_ * reconstruction_loss + beta_ * temporal_consistency_loss + gamma_ * denoise_ASL_loss + regularization_term
    return loss

def convlstm_block(input_layer, num_filters):
    convlstm = ConvLSTM2D(num_filters, (3, 3), padding='same', return_sequences=True)(input_layer)
    norm = BatchNormalization()(convlstm)
    # norm = LayerNormalization()(convlstm)
    relu = Activation('relu')(norm)
    return relu

def save_as_nifti(image, output_path, original_nifti_path):
    original_nifti = nib.load(original_nifti_path)
    denoised_nifti = nib.Nifti1Image(image, original_nifti.affine, original_nifti.header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(denoised_nifti, output_path)


def predict_in_batches(model, input_data, num_subj, batch_size):
    predictions = []
    batch_size = min(batch_size, num_subj)
    for i in range(0, num_subj, batch_size):
        batch = input_data[i : min(i + batch_size, num_subj)]
        batch_predictions = model.predict(batch)
        predictions.append(batch_predictions)
    return np.concatenate(predictions, axis=0)

def save_as_nifti(image_data, output_path, original_nifti_path):
    if isinstance(image_data, nib.Nifti1Image):
        nib.save(image_data, output_path)
    else:
        original_nifti = nib.load(original_nifti_path)
        new_img = nib.Nifti1Image(image_data, original_nifti.affine)
        nib.save(new_img, output_path)

def main():
    
    train_ratio = 0.7
    # validation_ratio = 0.2
    # test_ratio = 0.1
    
    input_data = [chunk for path in SUBIDS for chunk in get_nifti_data(path,target=0,overlap=overlap)]
    input_data = np.reshape(input_data, (-1, num_timepoint, chunk_size, length, 1))
    print("input", np.shape(input_data))

    target_data = [chunk for path in SUBIDS for chunk in get_nifti_data(path,target=1,overlap=overlap)]
    target_data = np.reshape(target_data, (-1, 1, chunk_size, length, 1))
    print("target", np.shape(target_data))

    X_train, X_val, y_train, y_val = train_test_split(input_data, target_data, test_size= 1 - train_ratio, shuffle=False)
    
    input_shape = exp_shape
    
    # (None, 168, 91, 109, 1)
    input_layer = Input(input_shape)
    
    block1 = convlstm_block(input_layer, 32)
    final = ConvLSTM2D(1, (3, 3), padding='same', return_sequences=True)(block1)

    print("X_train",np.shape(X_train))
    print("X_val",np.shape(X_val))
    print("y_train",np.shape(y_train))
    print("y_val",np.shape(y_val))

    training_model = Model(inputs=[input_layer], outputs=[final])
    training_model.summary()

    training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=custom_loss(training_model=training_model))  # Reduced learning rate
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    training_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[early_stopping])
    
    test_subjs=['MCI1036_042122']
    
    num_subj = len(test_subjs)
    for path in test_subjs:
        denoised_image_chunks = []
        input_chunk = get_nifti_data(path,overlap=0)
        denoised_chunk = predict_in_batches(training_model, input_chunk, chunk_size, chunk_size*num_subj)
        denoised_image_chunks.append(denoised_chunk)
        denoised_image = reassemble_image(denoised_image_chunks, overlap)
        denoised_image = np.squeeze(denoised_image)
        denoised_image = np.transpose(denoised_image, (2, 3, 0, 1))

        orig_path = '/data/GE_MCI/' + path + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss.nii.gz'
        original_nifti = nib.load(orig_path)
        affine = original_nifti.affine
        save_as_nifti(denoised_image, '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/denn_new/' + path + '_denoised.nii.gz', orig_path)
        
        # tsnr
        mean = np.mean(denoised_image, axis=3)
        std = np.std(denoised_image, axis=3)
        tsnr = mean / std
        
        tsnr_img = nib.Nifti1Image(tsnr, affine)
        save_as_nifti(tsnr_img, '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/denn_new/' + path + '_tsnr.nii.gz', orig_path)
        
        # mean
        mean_img = nib.Nifti1Image(mean, affine)
        save_as_nifti(mean_img, '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/denn_new/' + path + '_mean.nii.gz', orig_path)

if __name__ == "__main__":
    main()
