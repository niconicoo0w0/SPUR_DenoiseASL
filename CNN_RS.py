import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, ConvLSTM2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import nibabel as nib
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


#  try training on more subjects (~13)
SUBIDS=['GE_04', 'GE_06', 'GE_08', 'GE_10', 'GE_12', 'GE_26', 'GE_31', 
        'GE_32', 'GE_33', 'GE_36', 'GE_05', 'GE_07', 'GE_09', 'GE_11', 
        'GE_13',]
#  test on remaining subjects (~15)
TEST_SUBJECT=['GE_15','GE_14', 'GE_16','GE_18', 'GE_20', 'GE_22',
              'GE_17', 'GE_19', 'GE_21', 'GE_23',
              'GE_27', 'GE_29', 'GE_30']

index_to_split = 0
chunk_size = 91
length = 109
num_timepoint = 97

def get_nifti_data(subj, target=0, lstm=0):
    
    # target path: /data/GE_MCI/MCI_means/"
    if target == 1:
            nifti_file = nib.load('/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/ASL/RS/' + subj + '_GE_MBME_RS_ASL_s4_mean.nii.gz')
    else:
        nifti_file = nib.load('/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/ASL/RS/' + subj + '_GE_MBME_RS_ASL_s4.nii.gz')
    
    if lstm == 0:
        data = nifti_file.get_fdata()
        mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
        std = np.std(data, axis=(0, 1, 2), keepdims=True)
        data = (data - mean) / std
        if data.ndim == 3:
            data = np.expand_dims(data, axis=-1)
        data = np.expand_dims(data, axis=0)
    
    else:
        data = nifti_file.get_fdata()
        if data.ndim == 3:            
            mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
            std = np.std(data, axis=(0, 1, 2), keepdims=True)
            data = (data - mean) / (std + 1e-6)
            data = np.expand_dims(data, axis=1)
            data = np.transpose(data, (3, 1, 0, 2))
        else:
            mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
            std = np.std(data, axis=(0, 1, 2), keepdims=True)
            data = (data - mean) / (std + 1e-6)
            data = np.transpose(data, (2, 3, 0, 1))

    return data

def custom_loss():
    def perceptual_loss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    def loss(y_true, y_pred):
        alpha_ = 0.5
        gamma_ = 0.5
        
        reconstruction_loss = K.mean(K.square(y_true - y_pred))
        perceptual_loss_val = perceptual_loss(y_true, y_pred)
        return alpha_ * reconstruction_loss + gamma_ * perceptual_loss_val
    return loss

def custom_loss_rnn(training_model):
    
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

def conv_block(x, filters):
    x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x

def convlstm_block(input_layer, num_filters):
    convlstm = ConvLSTM2D(num_filters, (3, 3), padding='same', return_sequences=True)(input_layer)
    norm = BatchNormalization()(convlstm)
    relu = Activation('relu')(norm)
    return relu

def save_as_nifti(denoised_image, output_path1, output_path2, original_nifti_path):
    original_nifti = nib.load(original_nifti_path)
    # save orig img
    os.makedirs(os.path.dirname(output_path1), exist_ok=True)
    nib.save(original_nifti, output_path1)
    denoised_nifti = nib.Nifti1Image(denoised_image, original_nifti.affine, original_nifti.header)
    # save denoised img
    os.makedirs(os.path.dirname(output_path2), exist_ok=True)
    nib.save(denoised_nifti, output_path2)

def predict_in_batches(model, input_data, num_subj, batch_size):
    predictions = []
    batch_size = min(batch_size, num_subj)
    for i in range(0, num_subj, batch_size):
        batch = input_data[i : min(i + batch_size, num_subj)]
        batch_predictions = model.predict(batch)
        predictions.append(batch_predictions)
    return np.concatenate(predictions, axis=0)

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=0, *args, **kwargs):
        super(CustomEarlyStopping, self).__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            super(CustomEarlyStopping, self).on_epoch_end(epoch, logs)

def main():
    
    train_ratio = 0.7
    
    input_data = [chunk for path in SUBIDS for chunk in get_nifti_data(path,target=0)]
    input_data = np.reshape(input_data, (-1, chunk_size, length, chunk_size, num_timepoint))
    print("input", np.shape(input_data))

    target_data = [chunk for path in SUBIDS for chunk in get_nifti_data(path,target=1)]
    target_data = np.reshape(target_data, (-1, chunk_size, length, chunk_size, 1))
    print("target", np.shape(target_data))

    X_train, X_val, y_train, y_val = train_test_split(input_data, target_data, test_size=1-train_ratio, shuffle=False) 
    
    num_subj = len(SUBIDS)
    input_shape = (num_subj, chunk_size, length, chunk_size, num_timepoint)     #(6, 52, 62, 52, 97)
    input_layer = Input(input_shape[1:])                                        #(None, 52, 62, 52, 97)
    
    block1 = conv_block(input_layer, 32)
    final = Conv3D(num_timepoint, (3, 3, 3), padding='same')(block1)

    print("X_train",np.shape(X_train))
    print("X_val",np.shape(X_val))
    print("y_train",np.shape(y_train))
    print("y_val",np.shape(y_val))

    training_model = Model(inputs=[input_layer], outputs=[final])
    training_model.summary()

    training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=custom_loss())  # Reduced learning rate
    training_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=2)
    
    denoised_sequences = [training_model.predict(get_nifti_data(subj)) for subj in SUBIDS]
    denoised_sequences = np.array(denoised_sequences)
    # denoised_sequences = np.squeeze(np.array(denoised_sequences))
    # (15, 52, 62, 52, 97) => (780, 97, 52, 62, 1)
    denoised_sequences = np.reshape(denoised_sequences, (-1, num_timepoint, chunk_size, length, 1))
    print(np.shape(denoised_sequences))
    
    rnn_input_shape = (num_timepoint, chunk_size, length, 1)
    rnn_input_layer = Input(rnn_input_shape)
    print(np.shape(rnn_input_layer))
    rnn_model = convlstm_block(rnn_input_layer, 32)
    final = ConvLSTM2D(1, (3, 3), padding='same', return_sequences=True)(rnn_model)
    training_model_lstm = Model(inputs=[rnn_input_layer], outputs=[final])
    training_model_lstm.summary()
    training_model_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=custom_loss_rnn(training_model=training_model_lstm))
    target_sequences = [get_nifti_data(subj, target=1, lstm=1) for subj in SUBIDS]
    target_sequences = np.reshape(target_sequences, (-1, 1, chunk_size, length, 1))
    # （780, 1, 52, 62, 1)
    print(np.shape(target_sequences))
    
    early_stopping = CustomEarlyStopping(monitor='val_loss', patience=2, verbose=1, start_epoch=8)
    training_model_lstm.fit(denoised_sequences, target_sequences, epochs=15, batch_size=16, callbacks=[early_stopping])
    test_subjs=TEST_SUBJECT
    
    num_subj = len(test_subjs)
    for subj in test_subjs:
        orig_path = '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/ASL/RS/' + subj + '_GE_MBME_RS_ASL_s4.nii.gz'

        input_chunk = get_nifti_data(subj,lstm=1)
        denoised_image = training_model_lstm.predict(input_chunk)
        denoised_image = np.squeeze(denoised_image)
        print(np.shape(denoised_image))
        #(52, 97, 52, 62)
        denoised_image = np.reshape(denoised_image, (chunk_size, num_timepoint, chunk_size, length))
        denoised_image = np.transpose(denoised_image, (2, 3, 0, 1))
        
        # output orig path
        path1 = '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/ASL/RS/update2/' + subj + '_orig.nii.gz'
        # output denoised path
        path2 = '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/ASL/RS/update2/' + subj + '_denoised.nii.gz'
        
        save_as_nifti(denoised_image, path1, path2, orig_path)

if __name__ == "__main__":
    main()
