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
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter
from scipy import stats
from scipy.ndimage import generic_filter

SUBIDS = ['GE_04', 'GE_05', 'GE_06', 'GE_07', 'GE_08', 'GE_09', 'GE_10', 'GE_11', 'GE_12', 'GE_13', 'GE_14', 'GE_15', 'GE_16', 'GE_17', 'GE_18', 'GE_19', 'GE_20', 'GE_21', 'GE_22', 'GE_23', 'GE_26', 'GE_27', 'GE_29', 'GE_30', 'GE_31', 'GE_32', 'GE_33', 'GE_36']

DATA_SIZE = 52
LENGTH = 62
NUM_TIMEPOINT = 86
NUM_SUBJ = len(SUBIDS)
OVERLAP = 0  # Overlap size

exp_shape = (NUM_TIMEPOINT, DATA_SIZE, LENGTH, 1)

def get_nifti_data(subj, DATA_SIZE=DATA_SIZE, OVERLAP=0, target=0):
    
    if target == 1:
        nifti_file = nib.load('/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/ASL/BH/' + subj + '_GE_MBME_BH_ASL_s4_rs35_mean.nii.gz')
    else:
        nifti_file = nib.load('/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/ASL/BH/' + subj + '_GE_MBME_BH_ASL_s4_rs35.nii.gz')

    #nifti_file = nib.load(path)
    data = nifti_file.get_fdata()
    data = gaussian_filter(data, sigma=1)
    
    if data.ndim == 4:
        z_scores = stats.zscore(data, axis=None)
        abs_z_scores = np.abs(z_scores)
        outlier_mask = (abs_z_scores >= 3)
        data[outlier_mask] = np.nan

        def nanmean_kernel(values):
            return np.nanmean(values)

        data = generic_filter(data, nanmean_kernel, size=(3, 3, 3, 3))
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            global_mean = np.nanmean(data)
            data[nan_mask] = global_mean

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
    return data

def reassemble_image(image_chunks, OVERLAP):
    num_chunks = len(image_chunks)
    DATA_SIZE = image_chunks[0].shape[2]
    total_size = num_chunks * (DATA_SIZE - OVERLAP) + OVERLAP
    reassembled_image = np.zeros((DATA_SIZE, NUM_TIMEPOINT, total_size, LENGTH, 1)) # added a dimension here
    counts = np.zeros((DATA_SIZE, NUM_TIMEPOINT, total_size, LENGTH, 1))  # added a dimension here
    for i, chunk in enumerate(image_chunks):
        start = i * (DATA_SIZE - OVERLAP)
        end = start + DATA_SIZE
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
        gamma_ = 1.3
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
    relu = Activation('relu')(norm)
    return relu

def predict_in_batches(model, input_data, NUM_SUBJ, batch_size):
    predictions = []
    batch_size = min(batch_size, NUM_SUBJ)
    for i in range(0, NUM_SUBJ, batch_size):
        batch = input_data[i : min(i + batch_size, NUM_SUBJ)]
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

# fix early stopping
class CustomEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=0, *args, **kwargs):
        super(CustomEarlyStopping, self).__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            super(CustomEarlyStopping, self).on_epoch_end(epoch, logs)

def main():
    
    for idx, test_subj in enumerate(SUBIDS):

        print(f"Testing on {test_subj}")
        train_subj = [subj for j, subj in enumerate(SUBIDS) if j != idx]

        input_data_train = [chunk for path in train_subj for chunk in get_nifti_data(path, target=0, OVERLAP=OVERLAP)]
        input_data_train = np.reshape(input_data_train, (-1, NUM_TIMEPOINT, DATA_SIZE, LENGTH, 1))

        target_data_train = [chunk for path in train_subj for chunk in get_nifti_data(path, target=1, OVERLAP=OVERLAP)]
        target_data_train = np.reshape(target_data_train, (-1, 1, DATA_SIZE, LENGTH, 1))

        input_data_test = get_nifti_data(test_subj, target=0, OVERLAP=OVERLAP)
        input_data_test = np.reshape(input_data_test, (-1, NUM_TIMEPOINT, DATA_SIZE, LENGTH, 1))

        target_data_test = get_nifti_data(test_subj, target=1, OVERLAP=OVERLAP)
        target_data_test = np.reshape(target_data_test, (-1, 1, DATA_SIZE, LENGTH, 1))
  
        print(f"Making predictions for {test_subj}")
        model_path = f"model_{test_subj}.h5"
        
        trained_model = load_model(model_path, compile=False)
        trained_model.compile(loss=custom_loss(training_model=trained_model))
        
        denoised_image_chunks = []
        input_chunk = get_nifti_data(test_subj,OVERLAP=0)
        print(np.shape(input_chunk))
        
        denoised_chunk = predict_in_batches(trained_model, input_chunk, DATA_SIZE, DATA_SIZE*NUM_SUBJ)
        denoised_image_chunks.append(denoised_chunk)
        denoised_image = reassemble_image(denoised_image_chunks, OVERLAP)
        denoised_image = np.squeeze(denoised_image)
        denoised_image = np.transpose(denoised_image, (2, 3, 0, 1))

        orig_path = '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/ASL/BH/' + test_subj + '_GE_MBME_BH_ASL_s4_rs35.nii.gz'
        save_as_nifti(denoised_image, '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/ASL/BH/Result4/' + test_subj + '_denoised.nii.gz', orig_path)

if __name__ == "__main__":
    main()
