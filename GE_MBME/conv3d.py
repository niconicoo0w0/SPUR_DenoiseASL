import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, LayerNormalization, Conv3D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
import nibabel as nib
import scipy as scp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from tensorflow.keras.layers import Conv3D, MaxPooling3D, concatenate, UpSampling3D, Cropping3D


# SUBIDS=['GE_04','GE_18','GE_10','GE_12','GE_30','GE_19','GE_06']
SUBIDS=['GE_04', 'GE_05', 'GE_06', 'GE_07', 'GE_08', 'GE_09', 'GE_10', 'GE_11', 'GE_12', 'GE_13', 'GE_14', 'GE_15', 'GE_16', 'GE_17', 'GE_18', 'GE_19', 'GE_20', 'GE_21', 'GE_22', 'GE_23', 'GE_26', 'GE_27', 'GE_29', 'GE_30', 'GE_31', 'GE_32', 'GE_33', 'GE_36']

index_to_split = 2
chunk_size = 91
length = 109
num_timepoint = 99
overlap = 0  # Overlap size

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std

def denormalize(data, mean, std):
    return (data * std) + mean

def get_nifti_data(subj, target=0):
    if target == 1:
        nifti_file = nib.load('/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/' + subj + '_MBME_PCASL_RS_hp200_s4_mean.nii.gz')
    else:
        nifti_file = nib.load('/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/' + subj + '_MBME_PCASL_RS_hp200_s4.nii.gz')

    data = nifti_file.get_fdata()
    
    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)
    data = np.expand_dims(data, axis=0)
    return data

# better!
def conv_block(x, filters):
    x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x

def save_as_nifti(image, output_path, original_nifti_path):
    original_nifti = nib.load(original_nifti_path)
    denoised_nifti = nib.Nifti1Image(image, original_nifti.affine, original_nifti.header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(denoised_nifti, output_path)
    
def predict_in_batches(model, input_data, batch_size=32):
    predictions = []
    num_data_points = input_data.shape[0]
    for i in range(0, num_data_points, batch_size):
        batch = input_data[i : min(i + batch_size, num_data_points)]
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
    train_ratio = 0.7    
    
    input_data = [get_nifti_data(path, target=0) for path in SUBIDS]
    input_data = np.concatenate(input_data, axis=0)
    
    target_data = [get_nifti_data(path, target=1) for path in SUBIDS]
    target_data = np.concatenate(target_data, axis=0)

    X_train, X_val, y_train, y_val = train_test_split(input_data, target_data, test_size=1-train_ratio, shuffle=False) 
    X_train, train_mean, train_std = normalize(X_train)
    X_val = (X_val - train_mean) / train_std
    y_train, train_target_mean, train_target_std = normalize(y_train)
    y_val = (y_val - train_target_mean) / train_target_std
    
    num_subj = len(SUBIDS)
    input_shape = (num_subj, chunk_size, length, chunk_size, num_timepoint)     #(6, 91, 109, 91, 99)
    input_layer = Input(input_shape[1:])                                        #(None, 91, 109, 91, 99)

    block1 = conv_block(input_layer, 32)
    block2 = conv_block(block1, 64)
    final = Conv3D(num_timepoint, (3, 3, 3), padding='same')(block2)

    training_model = Model(inputs=[input_layer], outputs=[final])
    training_model.summary()
    
    training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mean_squared_error")
    
    
    early_stopping = CustomEarlyStopping(monitor='val_loss', patience=3, verbose=1, start_epoch=15)
    training_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=2, callbacks=[early_stopping])

    test_subjs=['GE_04', 'GE_05', 'GE_06', 'GE_07', 'GE_08', 'GE_09', 'GE_10', 'GE_11', 'GE_12', 'GE_13', 'GE_14', 'GE_15', 'GE_16', 'GE_17', 'GE_18', 'GE_19', 'GE_20', 'GE_21', 'GE_22', 'GE_23', 'GE_26', 'GE_27', 'GE_29', 'GE_30', 'GE_31', 'GE_32', 'GE_33', 'GE_36']

    num_subj = len(test_subjs)
    for path in test_subjs:     
        denoised_image_chunks = []
        input_chunk = get_nifti_data(path)
        print(np.shape(input_chunk))
        
        denoised_chunk = predict_in_batches(training_model, input_chunk)
        # denormalize
        denoised_image_chunks.append(denoised_chunk)
        denoised_image = np.squeeze(denoised_chunk)
        
        denoised_image = denormalize(denoised_image, train_mean, train_std)
        orig_path = '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/' + path + '_MBME_PCASL_RS_hp200_s4.nii.gz'
        save_as_nifti(denoised_image, '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/Result3/' + path + '_denoised.nii.gz', orig_path)

if __name__ == "__main__":
    main()
