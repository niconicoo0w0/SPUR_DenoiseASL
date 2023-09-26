import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, LayerNormalization, Conv3D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import nibabel as nib
import scipy as scp
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()


# SUBIDS=['GE_04','GE_18','GE_10','GE_12','GE_30','GE_19','GE_06']
# SUBIDS=['GE_04','GE_18','GE_10','GE_12','GE_30','GE_19','GE_06','GE_07','GE_08','GE_09','GE_17']
SUBIDS=['GE_04', 'GE_05', 'GE_06', 'GE_07', 'GE_08', 'GE_09', 'GE_10', 'GE_11', 'GE_12', 'GE_13', 'GE_14', 'GE_15', 'GE_16', 'GE_17', 'GE_18', 'GE_19', 'GE_20', 'GE_21', 'GE_22', 'GE_23', 'GE_26', 'GE_27', 'GE_29', 'GE_30', 'GE_31', 'GE_32', 'GE_33', 'GE_36']

index_to_split = 2
chunk_size = 91
length = 109
num_timepoint = 99
overlap = 0  # Overlap size

def get_nifti_data(subj, target=0):
    if target == 1:
        nifti_file = nib.load('/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/' + subj + '_MBME_PCASL_RS_hp200_s4_mean.nii.gz')
    else:
        nifti_file = nib.load('/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/' + subj + '_MBME_PCASL_RS_hp200_s4.nii.gz')

    data = nifti_file.get_fdata()
    mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
    std = np.std(data, axis=(0, 1, 2), keepdims=True)
    data = (data - mean) / std
    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)
    data = np.expand_dims(data, axis=0)
    return data

def conv_block(input_layer, num_filters):
    conv = Conv3D(num_filters, (3, 3, 3), padding='same')(input_layer)
    norm = BatchNormalization()(conv)
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

# loss function: PSNR
def psnr_loss(y_true, y_pred):
    max_pixel = 1.0

    mse = K.mean(K.square(y_true - y_pred))
    mse = K.maximum(mse, K.epsilon())

    psnr = 10.0 * K.log((max_pixel ** 2) / mse) / K.log(10.0)
    return -psnr

# # loss function: tSNR + MAE
# def tSNR(y_true, y_pred):
#     mean_signal = K.mean(y_true, axis=-1)  # mean over time axis
#     noise = y_true - y_pred
#     stddev_noise = K.std(noise, axis=-1)
#     return mean_signal / (stddev_noise + K.epsilon())

# def custom_loss(y_true, y_pred):
#     mae = K.mean(K.abs(y_true - y_pred))
#     ts = tSNR(y_true, y_pred)
#     return mae - ts

def main():
    num_subj = len(SUBIDS)
    trained_models = []
    
    for idx, test_subj in enumerate(SUBIDS):
        
        print(f"Testing on {test_subj}")
        train_subj = [subj for j, subj in enumerate(SUBIDS) if j != idx]

        input_data_train = [chunk for path in test_subj for chunk in get_nifti_data(path,target=0)]
        input_data_train = np.stack(input_data_train, axis=0)
        print("input", np.shape(input_data_train))
        
        target_data_train = [chunk for path in train_subj for chunk in get_nifti_data(path,target=1)]
        target_data_train = np.stack(target_data_train, axis=0)
        print("target", np.shape(target_data_train))
        
        input_data_test = get_nifti_data(test_subj, target=0, overlap=overlap)
        input_data_test = np.reshape(input_data_test, (num_subj, chunk_size, num_timepoint, chunk_size, length))

        target_data_test = get_nifti_data(test_subj, target=1, overlap=overlap)
        target_data_test = np.reshape(target_data_test, (num_subj, chunk_size, num_timepoint, chunk_size, length))
        
        input_shape = (num_subj, chunk_size, length, chunk_size, num_timepoint)     #(6, 91, 109, 91, 99)
        input_layer = Input(input_shape[1:])                                        #(None, 91, 109, 91, 99)

        block1 = conv_block(input_layer, 64)
        final = Conv3D(num_timepoint, (3, 3, 3), padding='same')(block1)

        training_model = Model(inputs=[input_layer], outputs=[final])
        training_model.summary()

        training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=psnr_loss)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        training_model.fit(input_data_train, target_data_train, validation_data=(input_data_test, target_data_test), epochs=50, batch_size=2, callbacks=[early_stopping])
        
        # save the result to trained models
        trained_models.append(training_model)
    
    test_subjs=['GE_04', 'GE_05', 'GE_06', 'GE_07', 'GE_08', 'GE_09', 'GE_10', 'GE_11', 'GE_12', 'GE_13', 'GE_14', 'GE_15', 'GE_16', 'GE_17', 'GE_18', 'GE_19', 'GE_20', 'GE_21', 'GE_22', 'GE_23', 'GE_26', 'GE_27', 'GE_29', 'GE_30', 'GE_31', 'GE_32', 'GE_33', 'GE_36']

    for idx, path in enumerate(test_subjs):     
        training_model = trained_models[idx]
        denoised_image_chunks = []
        input_chunk = get_nifti_data(path)
        print(np.shape(input_chunk))
        
        denoised_chunk = predict_in_batches(training_model, input_chunk, chunk_size, chunk_size*num_subj)
        denoised_image_chunks.append(denoised_chunk)
        denoised_image = np.squeeze(denoised_chunk)
        
        orig_path = '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/' + path + '_MBME_PCASL_RS_hp200_s4.nii.gz'
        save_as_nifti(denoised_image, '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/Result3/' + path + '_denoised.nii.gz', orig_path)

if __name__ == "__main__":
    main()
