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
        #data=np.mean(data,3)
        nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss_mean.nii.gz')
        #data = scp.stats.zscore(nifti_file.get_fdata())
    elif target == 2:
        nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/ROIs/wmparc.2.nii.gz')
      #  data = nifti_file.get_fdata()
    else:
        nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss.nii.gz')
        data = scp.stats.zscore(nifti_file.get_fdata(),axis=3)

    #nifti_file = nib.load(path)
    data = nifti_file.get_fdata()
    limit = total_size
    if data.ndim == 4:
        median = np.median(data, axis=(0, 1, 2), keepdims=True)
        iqr = scp.stats.iqr(data, axis=(0, 1, 2), keepdims=True)
        iqr[iqr < 1e-7] = 1
        data = (data - median) / iqr
        data = np.transpose(data, (2, 3, 0, 1))
        # mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
        # std = np.std(data, axis=(0, 1, 2), keepdims=True)
        # std[std < 1e-7] = 1
        # data = (data - mean) / std
        # data = np.transpose(data, (2, 3, 0, 1))
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=1)
        data = np.repeat(data, num_timepoint, axis=1)
        median = np.median(data, axis=(0, 2, 3), keepdims=True)
        iqr = scp.stats.iqr(data, axis=(0, 2, 3), keepdims=True)
        iqr[iqr < 1e-7] = 1
        data = (data - median) / iqr
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
    print(np.shape(image_chunks))
    num_chunks = len(image_chunks)
    chunk_size = image_chunks[0].shape[2]
    total_size = num_chunks * (chunk_size - overlap) + overlap
    reassembled_image = np.zeros((chunk_size, num_timepoint, total_size, length, 1)) # added a dimension here
    counts = np.zeros((chunk_size, num_timepoint, total_size, length, 1))  # added a dimension here
    for i, chunk in enumerate(image_chunks):
        print(i)
        start = i * (chunk_size - overlap)
        end = start + chunk_size
        reassembled_image[:, :, start:end, :, :] += chunk  # adjusted slicing here
        counts[:, :, start:end, :, :] += 1  # adjusted slicing here
    return reassembled_image / counts


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        x, (y_target, y_anat) = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            mse_loss = self.compiled_loss(y_target, y_pred, regularization_losses=self.losses)
            corr_loss = calculate_correlation(y_anat, y_pred)
            total_loss = 0.2 * mse_loss + 0.8 * corr_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y_target, y_pred)
        return {m.name: m.result() for m in self.metrics}


def calculate_correlation(y_true, y_pred, sample_weight=None):
   # assert len(y_true.shape)==5
    y_true = tf.transpose(y_true,[1,2,3,0,4])
    y_pred = tf.transpose(y_pred,[1,2,3,0,4])
   
   # tf.print(tf.reduce_max(y_true))

    c1T1_mask = tf.logical_and(tf.math.greater(y_true[0,:,:,:,:],1000),tf.math.less(y_true[0,:,:,:,:],3000))
    c23T1_mask = tf.logical_or(tf.logical_and(tf.math.less(y_true[0,:,:,:,:],1000),tf.math.greater(y_true[0,:,:,:,:],0)),tf.math.greater(y_true[0,:,:,:,:],3000))
    c1T1 = tf.boolean_mask(y_pred,c1T1_mask,axis=1)
    c23T1 = tf.boolean_mask(y_pred,c23T1_mask,axis=1)

    mean_ytrue = tf.reduce_mean(c1T1, keepdims=True, axis=[1])
    mean_ypred = tf.reduce_mean(c23T1, keepdims=True, axis=[1])

    correlation=tfp.stats.correlation(mean_ytrue,mean_ypred)

    return tf.math.abs(tf.maximum(tf.minimum(correlation, 1.0), -1.0))

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

def main():
    parser = argparse.ArgumentParser(description='Train CNN with ASL data.')
    #parser.add_argument('--data', nargs='+', type=str, help='List of paths to the ASL data.')
    #parser.add_argument('--target', nargs='+', type=str, help='List of paths to the target (gold standard) data.')
    parser.add_argument('--output', type=str, help='Path to save the denoised data.')

    train_ratio=0.7

    input_data = [chunk for path in SUBIDS for chunk in get_nifti_data(path,target=0,overlap=overlap)]
    input_data = np.reshape(input_data, (-1, num_timepoint, chunk_size, length, 1))
    print("input", np.shape(input_data))

    target_data = [chunk for path in SUBIDS for chunk in get_nifti_data(path,target=1,overlap=overlap)]
    target_data = np.reshape(target_data, (-1, num_timepoint, chunk_size, length, 1))
    print("target", np.shape(target_data))

    anat_data = [chunk for path in SUBIDS for chunk in get_nifti_data(path,target=2,overlap=overlap)]
    anat_data = np.reshape(anat_data, (-1, num_timepoint, chunk_size, length, 1))
    print("anat", np.shape(anat_data))

    data_indices = np.any(anat_data > 3000,axis=(1,2,3,4))
    target_data = target_data[data_indices,:]
    anat_data = anat_data[data_indices,:]
    input_data = input_data[data_indices,:]
    indices=np.arange(input_data.shape[0])

    X_train, X_val, _, _ = train_test_split(input_data,
                                            indices,
                                            test_size=1 - train_ratio,
                                            shuffle=False)
    print("X_train",np.shape(X_train))
    print("X_val",np.shape(X_val))

    anat_train, anat_val = train_test_split(anat_data,
                                      test_size=1 - train_ratio,
                                      shuffle=False)
    print("anat_train",np.shape(anat_train))
    print("anat_val",np.shape(anat_val))
    
    target_train, target_val = train_test_split(target_data,
                                      test_size=1 - train_ratio,
                                      shuffle=False)
    print("target_train",np.shape(target_train))
    print("target_val",np.shape(target_val))

    input_shape = exp_shape
    
    # (None, 168, 52, 62, 1)
    input_layer = Input(input_shape)
    
    block1 = convlstm_block(input_layer, 8)
    final = ConvLSTM2D(1, (3, 3), padding='same', return_sequences=True)(block1)

    training_model = CustomModel(inputs=[input_layer], outputs=[final])
    training_model.summary()

    # training_model.compile(optimizer='adam', loss=custom_loss(training_model=training_model))
    training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.MeanSquaredError()) 
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    training_model.fit(X_train, [target_train, anat_train], batch_size=5, epochs=10, validation_data=(X_val, [target_val, anat_val]), callbacks=[es], verbose=2)

    test_subjs=['MCI1036_042122']

    num_subj = len(test_subjs)
    #print(num_subj)
    for i, path in enumerate(test_subjs):
        denoised_image_chunks = []
        input_chunk = get_nifti_data(path,overlap=0)
        print(i, np.shape(input_chunk))
        denoised_chunk = predict_in_batches(training_model, input_chunk, 91, 91*num_subj)
        denoised_image_chunks.append(denoised_chunk)
        denoised_image = reassemble_image(denoised_image_chunks, overlap)
        denoised_image = np.squeeze(denoised_image)
        denoised_image = np.transpose(denoised_image, (2, 3, 0, 1))
        save_as_nifti(denoised_image, '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/anat_layer/' + path + '_denoised.nii.gz','/data/GE_MCI/' + path + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss.nii.gz')

if __name__ == "__main__":
    main()