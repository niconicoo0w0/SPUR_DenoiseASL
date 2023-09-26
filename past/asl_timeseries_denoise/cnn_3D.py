"""
Created on Dec 28 2019
Code for 3D task activation regression with convolutional networks based on resting state connectivity data
@author: mregina
"""

import tensorflow as tf
from datetime import datetime
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import NiiSequence, CorrelationMetric, CorrelationLoss, save_prediction

DATA_PATH = '/data/GE_MCI/'
OUT_PATH = 'Scripts/DenoiseASL/AlexTest/'
#with open('/media/Drobo_HCP/HCP_Data/Volume/CNN/list_train.txt') as f:
#	TRAIN_SUBIDS = f.read().split('\n')
#TRAIN_SUBIDS.pop()

#with open('/media/Drobo_HCP/HCP_Data/Volume/CNN/list_val.txt') as f:
#	VALID_SUBIDS = f.read().split('\n')
#VALID_SUBIDS.pop()

#with open('/media/Drobo_HCP/HCP_Data/Volume/CNN/list_test.txt') as f:
#	TEST_SUBIDS = f.read().split('\n')
#TEST_SUBIDS.pop()

with open(DATA_PATH + '/list_MBME_PCASL_RS.txt') as f:
    SUBIDS = f.read().split('\n')
SUBIDS.pop()

train_ratio = 0.5
validation_ratio = 0.3
test_ratio = 0.2

X_data, X_test, y_data, y_test = train_test_split(np.arange(np.shape(SUBIDS)[0]),SUBIDS, test_size=test_ratio, train_size=1-test_ratio)
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=validation_ratio/(1-test_ratio),train_size=train_ratio/(1-test_ratio))

print(y_train)
print(y_val)
print(y_test)

TRAIN_SUBIDS = ['MCI1007','MCI1008']
VALID_SUBIDS = ['MCI1033_062822']
TEST_SUBIDS = ['MCI1018_112321']
#/data/GE_MCI/MCI1007/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss.nii.gz
REST_FILE_NAME = 'HyperMEPI_PCASL_RS_PA_ASL'
TASK_FILE_NAME = 'HyperMEPI_PCASL_RS_PA_ASL'
TASK_FILE_NUM = '3'
# output paths
LOGDIR = os.path.join("logs/regress_3D", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(os.path.join(LOGDIR, "checkpoints"), exist_ok=False)
CHECKPOINT_PATH = os.path.join(LOGDIR, "checkpoints", "cp-{epoch:04d}.ckpt")

BATCH_SIZE = 3
INPUT_SHAPE = (91, 109, 91, 168)
OUTPUT_SIZE = (91, 109, 91, 168)

# hyperparameters
LEARNING_RATE = 0.001  ## Lower = slower but more accurate
EPOCH = 20
DROPOUT_RATE = None  # 0.1

# architecture parameters
filter_num1 = 64  # 96
filter_num2 = 32  # 64
filter_num3 = 16  # 32

# tf.config.experimental_run_functions_eagerly(True)

# create datasets
train_dataset = NiiSequence(y_train, shuffle=True, rootpath=DATA_PATH, dataname=REST_FILE_NAME,
                            labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM, batch_size=BATCH_SIZE)
valid_dataset = NiiSequence(y_val, rootpath=DATA_PATH, dataname=REST_FILE_NAME, labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM,
                            batch_size=BATCH_SIZE)
test_dataset = NiiSequence(y_test, rootpath=DATA_PATH, dataname=REST_FILE_NAME, labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM,
                            batch_size=BATCH_SIZE)

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

# create model
input = tf.keras.Input(shape=INPUT_SHAPE, name='input')

block1 = separable_convolution_3D(input, kernel_size=5, filter_num=filter_num1, name='block1_',
                                  activation='relu', dropout_rate=DROPOUT_RATE)
block2 = separable_convolution_3D(block1, kernel_size=3, filter_num=filter_num2, name='block2_',
                                  activation='relu', dropout_rate=DROPOUT_RATE)
block3 = separable_convolution_3D(block2, kernel_size=3, filter_num=filter_num3, name='block3_',
                                  activation='relu', dropout_rate=DROPOUT_RATE)

output = convolution_3D(block3, kernel_size=3, filter_num=168, name='block4_',
                                  activation=None, dropout_rate=DROPOUT_RATE)
model = tf.keras.Model(inputs=[input], outputs=[output])

# create callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=False, period=5)
stop_callback = tf.keras.callbacks.EarlyStopping(monitor=mean_squared_error, min_delta=0, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=True)
# compile model for training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              # Loss function to minimize
              loss=tf.keras.losses.MeanSquaredError(),
              # List of metrics to monitor
              metrics=[tf.keras.losses.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), CorrelationMetric()])
model.summary()

# save model architecture in json
model_json = model.to_json()
with open(LOGDIR + "/model.json", "w") as json_file:
    json_file.write(model_json)

# training loop
model.fit(train_dataset, epochs=EPOCH, validation_data=valid_dataset, callbacks=[tensorboard_callback, cp_callback, stop_callback])

# evaluate on the test set
test_loss = model.evaluate(test_dataset)
print(test_loss)

# predict on test data
predicted_batch = model.predict(test_dataset)
print(np.shape(predicted_batch))
save_prediction(predicted_batch=predicted_batch, rootpath=DATA_PATH, outpath=OUT_PATH, labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM,
                template_subID=y_test[0], subIDs=y_test)
