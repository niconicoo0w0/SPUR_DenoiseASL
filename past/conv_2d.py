#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from datetime import datetime
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils_lstm import CorrelationMetric
import nibabel as nib
from tensorflow.keras import backend as K
import scipy as scp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import keras.backend as T
import tensorflow_probability as tfp

DATA_PATH = '/data/GE_MCI/'
OUT_PATH = 'Scripts/DenoiseASL/LSTM/'
REST_FILE_NAME = 'HyperMEPI_PCASL_RS_PA_ASL'
TASK_FILE_NAME = 'HyperMEPI_PCASL_RS_PA_ASL'
TASK_FILE_NUM = '3'

with open(DATA_PATH + '/list_MBME_PCASL_RS.txt') as f:
    SUBIDS = f.read().split('\n')
SUBIDS.pop()
SUBIDS=['MCI1007','MCI1008','MCI1017','MCI1019_120121','MCI1020_112321','MCI1037_042122']

index_to_split = 2
chunk_size = 52
total_size = 52
num_timepoint = 168
overlap = 0  # Overlap size
exp_shape = (num_timepoint, chunk_size, 62, 1)


# In[27]:


class NiiSequence2D(tf.keras.utils.Sequence):
    def __init__(self, data, label, rootpath, dataname, labelname, labelnum, batch_size, shuffle=False):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.rootpath = rootpath
        self.dataname = dataname
        self.labelname = labelname
        self.labelnum = labelnum
        self.shuffle = shuffle

    def __len__(self):
        return np.ceil(len(self.data) / self.batch_size).astype(np.int64)

    def __getitem__(self, idx):
       # print(idx)
        if self.shuffle and idx == 0:
            shuffle_ids = np.arange(np.shape(self.data)[0])
            np.random.shuffle(shuffle_ids)
            self.data = self.data[shuffle_ids,:]
            self.label = self.label[shuffle_ids,:]

            #print(shuffle_ids)
      #  print(self.data.shape)
        #print(self.label.shape)
    #slices = np.array(self.subIDs)[shuffle_ids]
        data_batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size,:]
        label_batch = self.label[idx * self.batch_size:(idx + 1) * self.batch_size,:]
        #print(data_batch.shape)
        #print(label_batch.shape)
   # low = idx * self.batch_size
    #high = min(low + self.batch_size, len(self.x))
        return data_batch, label_batch
            #return np.stack(data_batch, axis=0), np.stack(label_batch, axis=0)


def separable_convolution_3D(x, kernel_size, filter_num, name, activation='relu', dropout_rate=None):
    conv1 = tf.keras.layers.Conv3D(filters=filter_num, kernel_size=(kernel_size, 1, 1), strides=(1, 1, 1),
                                   padding='same', kernel_initializer='he_uniform', activation=activation,
                                   name=name + 'conv1')(x)
    conv2 = tf.keras.layers.Conv3D(filters=filter_num, kernel_size=(1, kernel_size, 1), strides=(1, 1, 1),
                                   padding='same', kernel_initializer='he_uniform', activation=activation,
                                   name=name + 'conv2')(conv1)
    conv3 = tf.keras.layers.Conv3D(filters=filter_num, kernel_size=(1, 1, kernel_size), strides=(1, 1, 1),
                                   padding='same', kernel_initializer='he_uniform', activation=activation,
                                   name=name + 'conv3')(conv2)
    if dropout_rate is not None:
        conv3 = tf.keras.layers.SpatialDropout3D(rate=dropout_rate, name=name + 'dropout')(conv3)
    return conv3

def convolution_3D(x, kernel_size, filter_num, name, activation='relu', dropout_rate=None):
    conv1 = tf.keras.layers.Conv3D(filters=filter_num, kernel_size=(kernel_size, 1, 1), strides=(1, 1, 1),
                                   padding='same', kernel_initializer='he_uniform', activation=activation,
                                   name=name + 'conv1')(x)

    if dropout_rate is not None:
        conv1 = tf.keras.layers.SpatialDropout3D(rate=dropout_rate, name=name + 'dropout')(conv1)
    return conv1

def convolution_3D_LSTM(x, kernel_size, filter_num, name, activation='relu', dropout_rate=None):
    conv1 = tf.keras.layers.ConvLSTM3D(filters=filter_num, kernel_size=(kernel_size, 2, 2), strides=(1, 1, 1),
                                   padding='same', kernel_initializer='he_uniform', activation=activation,
                                   name=name + 'conv1',return_sequences=True)(x)

    if dropout_rate is not None:
        conv1 = tf.keras.layers.SpatialDropout3D(rate=dropout_rate, name=name + 'dropout')(conv1)
    return conv1

def convolution_2D_LSTM(x, kernel_size, filter_num, name, activation='relu', dropout_rate=None):
    conv1 = tf.keras.layers.ConvLSTM2D(filters=filter_num, kernel_size=(kernel_size, kernel_size),
                                   padding='same', activation=activation,
                                   name=name + 'conv1',return_sequences=True)(x)

    if dropout_rate is not None:
        conv1 = tf.keras.layers.SpatialDropout3D(rate=dropout_rate, name=name + 'dropout')(conv1)
    return conv1

def get_ts_data(subj):
    print(subj)
    nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss.nii.gz')
    return np.transpose(nifti_file.get_fdata(),(2,3,0,1))[45:55,:]

def get_target_data(subj):
    nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss_mean.nii.gz')
    return np.transpose(nifti_file.get_fdata(),(2,0,1))[45:55,:]

def get_test_data(subj):
    print(subj)
    nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss.nii.gz')
    return np.transpose(nifti_file.get_fdata(),(2,3,0,1))

def save_prediction(predicted_batch, rootpath, outpath, template_subID, labelname, labelnum, batch_id=None, subIDs=None, outsuf=""):
    #template_img =nibabel.load(rootpath + template_subID + '/task/tfMRI_' + labelname + '/tfMRI_' + labelname + '_hp200_s4_level2vol.feat/cope' + labelnum + '.feat/stats/tstat1.nii.gz').get_fdata()
    template_img = nib.load('/data/GE_MCI/MCI1007/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss_rs35.nii.gz')
    print(np.shape(template_img))
    print(predicted_batch.shape)
    batch_size = predicted_batch.shape[0]
    #for i in range(batch_size):
    print(np.transpose(predicted_batch,axes=(2,3,0,1,4)).shape)
    new_img = nib.Nifti1Image(np.transpose(predicted_batch,axes=(2,3,0,1,4)), template_img.affine, template_img.header)
    filename = rootpath + outpath + subIDs + '_predicted_' + labelname + outsuf + '.nii.gz'

    print(filename)
    nib.save(new_img, filename)

def custom_loss(training_model):
    def loss(y_true, y_pred):
        alpha_ = 1.0
        beta_ = 0.1
        gamma_ = 1.0
        lambda_ = 0.001
        reconstruction_loss = K.mean(K.square(y_true - y_pred))
        y_pred_right_shifted = K.concatenate([y_pred[:, 1:, :], y_pred[:, -1:, :]], axis=1)
        temporal_consistency_loss = K.mean(K.square(y_pred - y_pred_right_shifted))
        denoise_ASL_loss = denoise_loss_ASL(y_true, y_pred)
        regularization_term = lambda_ * K.sum([K.sum(K.square(w)) for w in training_model.trainable_weights])
        return alpha_ * reconstruction_loss + beta_ * temporal_consistency_loss + gamma_ * denoise_ASL_loss + regularization_term
    return loss

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

def get_nifti_data(subj, chunk_size=chunk_size, overlap=5, target=0):
   # nifti_file = nib.load(path)
    print(subj)
    limit = total_size
    
    if target == 1:
        #data=np.mean(data,3)
        nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss_mean_rs35.nii.gz')
        data = scp.stats.zscore(nifti_file.get_fdata())
    elif target == 2:
        nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/ROIs/wmparc.35.nii.gz')
        data = nifti_file.get_fdata()
    else:
        nifti_file = nib.load('/data/GE_MCI/' + subj + '/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss_rs35.nii.gz')
        data = scp.stats.zscore(nifti_file.get_fdata())

    #print(data.shape)
    limit = total_size
    #print(data.ndim)
    if data.ndim == 4:
      #  mean = np.mean(data, axis=(0, 1, 2, 3), keepdims=True)
      #  std = np.std(data, axis=(0, 1, 2, 3), keepdims=True)
      #  data = (data - mean) / (std + 1e-6)
        data = np.transpose(data, (2, 3, 0, 1))
    elif data.ndim == 3:
      #  mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
       # std = np.std(data, axis=(0, 1, 2), keepdims=True)
        #data = (data - mean) / (std + 1e-6)
        data = np.transpose(data, (2, 0, 1))
        data = np.expand_dims(data, axis=1)
       # if target == 1:
        data = np.repeat(data, num_timepoint, axis=1)
    for i in range(0, limit, chunk_size - overlap):
        chunk = data[:, :, i:min(i+chunk_size, limit), :]
        if chunk.shape[index_to_split] < chunk_size:
            padding_shape = list(chunk.shape)
            padding_shape[index_to_split] = chunk_size - chunk.shape[index_to_split]
            padding = np.zeros(padding_shape)
            chunk = np.concatenate([chunk, padding], axis=index_to_split)
        chunk = np.expand_dims(chunk, axis=-1)
        #print(np.shape(chunk))
       # yield chunk
        return np.nan_to_num(chunk[21:31,:])

def convlstm_block(input_layer, num_filters):
    convlstm = tf.keras.layers.ConvLSTM2D(num_filters, (3, 3), padding='same', return_sequences=True)(input_layer)
    #norm = tf.keras.layers.BatchNormalization()(convlstm)
    norm = tf.keras.layers.LayerNormalization()(convlstm)
    relu = tf.keras.layers.Activation('relu')(norm)
    return relu

def denoise_loss_wrapper(input_tensor):
    def denoise_loss(y_true,y_pred):
        print(tf.math.reduce_max(y_true))
        c1T1 = tf.logical_and(tf.math.greater(y_true,1000),tf.math.less(y_true,3000)).astype(float)*(y_true > 0).astype(np.float32)*y_pred.astype(float)
        c23T1 = tf.logical_or(tf.math.less(y_true,1000),tf.math.greater(y_true,3000)).astype(float)*(y_true > 0).astype(np.float32)*y_pred.astype(float)
        print(tf.math.reduce_max(c1T1))
        print(c1T1)

        output_fMRI = np.reshape(c1T1,(32240,168))
    #    output_dwt = y_true[:,:,0]
        output_dwt  = c23T1
        #tf.print(y_pred[10,10,:])
        tdim = 168
        output_fMRI = output_fMRI - T.mean(output_fMRI,axis=-1,keepdims=True)
        output_dwt  = output_dwt - T.mean(output_dwt,axis = -1,keepdims=True)
        output_fMRI = output_fMRI/T.std(output_fMRI,axis=-1,keepdims=True)
        output_dwt  = output_dwt/T.std(output_dwt,axis=-1,keepdims=True)
        print(output_fMRI)
       # corr_mat = T.dot(output_fMRI,T.transpose(output_fMRI))/tdim
       # corr_fMRI = T.mean(T.abs(corr_mat))/2
       # corr_mat = T.dot(output_dwt,T.transpose(output_dwt))/tdim
       # corr_dwt = T.mean(T.abs(corr_mat))/2    
        corr_mat = T.dot(output_fMRI,T.transpose(output_dwt))/tdim
        corr_fMRIdwt = T.mean(T.abs(corr_mat))
       # tf.print(corr_fMRIdwt)
        return corr_fMRIdwt #corr_dwt - corr_fMRI 
    return denoise_loss


# In[28]:


input_data = [chunk for path in SUBIDS for chunk in get_nifti_data(path,target=0,overlap=overlap)]
input_data = np.reshape(input_data, (-1, num_timepoint, chunk_size, 62, 1))
print("input", np.shape(input_data))

target_data = [chunk for path in SUBIDS for chunk in get_nifti_data(path,target=1,overlap=overlap)]
target_data = np.reshape(target_data, (-1, num_timepoint, chunk_size, 62, 1))
print("target", np.shape(target_data))

anat_data = [chunk for path in SUBIDS for chunk in get_nifti_data(path,target=2,overlap=overlap)]
print("anat", np.shape(anat_data))
anat_data = np.reshape(anat_data, (-1, num_timepoint, chunk_size, 62, 1))
print("anat", np.shape(anat_data))


# In[29]:


train_ratio = 0.5
validation_ratio = 0.25
test_ratio = 0.25

print(np.max(anat_data))

data_indices = np.any(target_data > 0,axis=(1,2,3,4))
target_data = target_data[data_indices,:]
anat_data = anat_data[data_indices,:]
input_data = input_data[data_indices,:]
indices=np.arange(input_data.shape[0])

print("input_data", np.shape(input_data))
print("target_data", np.shape(target_data))

X_data, X_test, y_data, y_test, indices_data, indices_test = train_test_split(input_data, anat_data, indices, test_size=1 - (train_ratio+validation_ratio))
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=validation_ratio/(train_ratio + validation_ratio))

print(np.max(y_train))

#print(y_train.shape)
#print(X_train.shape)

#print(y_val.shape)
#print(y_test.shape)
#print(indices_test)


# In[30]:


print("input_data", np.shape(X_train))
print("target_data", np.shape(y_train))
print(np.any(np.isnan(input_data)))


# In[35]:


# hyperparameters

LEARNING_RATE = 1e-3  ## Lower = slower but more accurate
EPOCH = 50
DROPOUT_RATE = None  # 0.1
BATCH_SIZE = 1
INPUT_SHAPE = exp_shape

# architecture parameters
filter_num1 = 32  # 96
filter_num2 = 8  # 64
filter_num3 = 8  # 32

# create datasets
train_dataset = NiiSequence2D(X_train,y_train, shuffle=True, rootpath=DATA_PATH, dataname=REST_FILE_NAME,
                            labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM, batch_size=BATCH_SIZE)
#print(train_dataset.__getitem__(0))
valid_dataset = NiiSequence2D(X_val,y_val, rootpath=DATA_PATH, dataname=REST_FILE_NAME, labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM,
                            batch_size=BATCH_SIZE)
test_dataset = NiiSequence2D(X_test,y_test, rootpath=DATA_PATH, dataname=REST_FILE_NAME, labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM,
                            batch_size=BATCH_SIZE)

LOGDIR = os.path.join(DATA_PATH + OUT_PATH + "logs/regress_3D", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(os.path.join(LOGDIR, "checkpoints"), exist_ok=False)
CHECKPOINT_PATH = os.path.join(LOGDIR, "checkpoints", "cp-{epoch:04d}.ckpt")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=False, save_freq='epoch')
stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min', baseline=None, restore_best_weights=True)


# In[38]:


class CorrelationLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred, sample_weight=None):
       # print(y_true.shape)
        correlation = calculate_correlation(y_true, y_pred, sample_weight)
        return correlation

def calculate_correlation(y_true, y_pred, sample_weight=None):
   # assert len(y_true.shape)==5

    y_true = tf.transpose(y_true,[1,2,3,0,4])
    y_pred = tf.transpose(y_pred,[1,2,3,0,4])
   
    c1T1_mask = tf.logical_and(tf.math.greater(y_true[1,:,:,:],1000),tf.math.less(y_true[1,:,:,:,:],3000))
    c23T1_mask = tf.logical_or(tf.logical_and(tf.math.less(y_true[1,:,:,:,:],1000),tf.math.greater(y_true[1,:,:,:,:],0)),tf.math.greater(y_true[1,:,:,:,:],3000))
    c1T1 = tf.boolean_mask(y_pred,c1T1_mask,axis=1)
    c23T1 = tf.boolean_mask(y_pred,c23T1_mask,axis=1)

    mean_ytrue = tf.reduce_mean(c1T1, keepdims=True, axis=[1])
    mean_ypred = tf.reduce_mean(c23T1, keepdims=True, axis=[1])

    correlation=tfp.stats.correlation(mean_ytrue,mean_ypred)

    return tf.math.abs(tf.maximum(tf.minimum(correlation, 1.0), -1.0))

# In[39]:


input = tf.keras.Input(shape=INPUT_SHAPE, name='input')
#block1 = convolution_2D_LSTM(input, kernel_size=3, filter_num=filter_num1, name='block1_',
   #                               activation='relu', dropout_rate=DROPOUT_RATE)
#block2 = convolution_3D_LSTM(block1, kernel_size=5, filter_num=filter_num2, name='block2_',
 #                                 activation='relu', dropout_rate=DROPOUT_RATE)
#output = convolution_2D_LSTM(block1, kernel_size=3, filter_num=1, name='output_',
   #                               activation='tanh', dropout_rate=DROPOUT_RATE)
block1 = convlstm_block(input, 32)
final = tf.keras.layers.ConvLSTM2D(1, (3, 3), padding='same', return_sequences=True)(block1)
model = tf.keras.Model(inputs=[input], outputs=[final])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              # Loss function to minimize
              loss=CorrelationLoss())
print(model.summary())


# In[40]:


# training loop
model.fit(train_dataset, epochs=EPOCH, validation_data=valid_dataset, use_multiprocessing=False,
          callbacks=[tensorboard_callback, cp_callback, stop_callback])


# In[42]:


test_id=['MCI1036_042122']
#[chunk for path in SUBIDS for chunk in get_nifti_data(path,target=0,overlap=overlap)]
test_data = np.asarray([data for path in test_id for data in get_nifti_data(path,target=0,overlap=overlap)])

#test_data = np.squeeze(test_data,axis=0)
print(np.shape(test_data))
#test_data=np.expand_dims(test_data, axis=-1)

#for i in np.arange(91):
#model.summary() 
predicted_batch = model.predict(test_data)

print(predicted_batch.shape)
save_prediction(predicted_batch=predicted_batch, rootpath=DATA_PATH, outpath=OUT_PATH, labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM,
                template_subID=y_test[0], subIDs=test_id[0],outsuf='_layernorm_filt32_rs35_ep50_batch1_denoiseloss')


# In[ ]:


save_prediction(predicted_batch=test_data, rootpath=DATA_PATH, outpath=OUT_PATH, labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM,
                template_subID=y_test[0], subIDs=test_id[0])


# In[ ]:





# In[ ]:


