"""
Created on Dec 29 2019
Code for 3D task activation regression with convolutional networks based on resting state connectivity data
@author: mregina
"""

import numpy as np
import tensorflow as tf
import nibabel

# correlation calculation for keras metric and loss classes
def calculate_correlation(y_true, y_pred, sample_weight=None):
    assert len(y_true.shape)==5
    mean_ytrue = tf.reduce_mean(y_true, keepdims=True, axis=[1,2,3,4])
    mean_ypred = tf.reduce_mean(y_pred, keepdims=True, axis=[1,2,3,4])

    demean_ytrue = tf.cast(y_true - mean_ytrue,tf.float32) #ADC added tf.cast to resolve type differences between demean_ytrue and demean_ypred
    demean_ypred = tf.cast(y_pred - mean_ypred,tf.float32) #ADC added tf.cast to resolve type differences between demean_ytrue and demean_ypred

    if sample_weight is not None:
        sample_weight = tf.broadcast_weights(sample_weight, y_true)
        std_y = tf.sqrt(tf.reduce_sum(sample_weight * tf.square(demean_ytrue)) * tf.reduce_sum(
            sample_weight * tf.square(demean_ypred)))
        correlation = tf.reduce_sum(sample_weight * demean_ytrue * demean_ypred) / std_y
    else:
        std_y = tf.sqrt(tf.reduce_sum(tf.square(demean_ytrue)) * tf.reduce_sum(tf.square(demean_ypred)))
        correlation = tf.reduce_sum(demean_ytrue * demean_ypred) / std_y
    return tf.maximum(tf.minimum(correlation, 1.0), -1.0)

# correlation metric
class CorrelationMetric(tf.keras.metrics.Metric):
    def __init__(self, name="correlation", **kwargs):
        super(CorrelationMetric, self).__init__(name, **kwargs)
        self.correlation = self.add_weight(name='correlation', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        correlation = calculate_correlation(y_true, y_pred, sample_weight)
        self.correlation.assign(correlation)

    def result(self):
        return self.correlation


# correlation as loss function
class CorrelationLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred, sample_weight=None):
        correlation = calculate_correlation(y_true, y_pred, sample_weight)
        return 1.0 - correlation


# create input datasets as a sequence
class NiiSequence(tf.keras.utils.Sequence):
    def __init__(self, subIDs, rootpath, dataname, labelname, labelnum, batch_size, shuffle=False):
        self.subIDs = subIDs
        self.batch_size = batch_size
        self.rootpath = rootpath
        self.dataname = dataname
        self.labelname = labelname
        self.labelnum = labelnum
        self.shuffle = shuffle

    def __len__(self):
        return np.ceil(len(self.subIDs) / self.batch_size).astype(np.int64)

    def __getitem__(self, idx):
        if self.shuffle and idx == 0:
            shuffle_ids = np.arange(len(self.subIDs))
            np.random.shuffle(shuffle_ids)
            self.subIDs = np.array(self.subIDs)[shuffle_ids]
        subID_batch = self.subIDs[idx * self.batch_size:(idx + 1) * self.batch_size]
        data_batch = []
        label_batch = []
        for subID in subID_batch:
            print(subID)
            data = nibabel.load(self.rootpath + subID + '/MNINonLinear/Results/' + self.dataname + '/' + self.dataname + '_hp2000_s6_level1.feat/' + self.dataname + '_hp2000_s6_e1_ss.nii.gz').get_fdata()
            label = nibabel.load(self.rootpath + subID + '/MNINonLinear/Results/' + self.dataname + '/' + self.dataname + '_hp2000_s6_level1.feat/' + self.dataname + '_hp2000_s6_e1_ss_mean.nii.gz').get_fdata()
            label = np.expand_dims(label, axis=3)

            data_batch.append(data)
            label_batch.append(label)
        if self.batch_size>1:
            return np.stack(data_batch, axis=0), np.stack(label_batch, axis=0)
        else:
            return np.expand_dims(data, axis=0), np.expand_dims(label, axis=0)

        
# save predicted images in niftii format for later tests and visual checking
def save_prediction(predicted_batch, rootpath, outpath, template_subID, labelname, labelnum, batch_id=None, subIDs=None):
    #template_img =nibabel.load(rootpath + template_subID + '/task/tfMRI_' + labelname + '/tfMRI_' + labelname + '_hp200_s4_level2vol.feat/cope' + labelnum + '.feat/stats/tstat1.nii.gz').get_fdata()
    template_img = nibabel.load('/data/GE_MCI/MCI1007/MNINonLinear/Results/HyperMEPI_PCASL_RS_PA_ASL/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_level1.feat/HyperMEPI_PCASL_RS_PA_ASL_hp2000_s6_e1_ss.nii.gz')
    print(np.shape(template_img))
    print(predicted_batch.shape)
    batch_size = predicted_batch.shape[0]
    for i in range(batch_size):
        print(i)
        new_img = nibabel.Nifti1Image(predicted_batch[i, :, :, :, :], template_img.affine, template_img.header)
        if subIDs is not None:
            filename = rootpath + outpath + subIDs[i] + '_predicted_' + labelname + labelnum + '.nii.gz'
        elif batch_id is not None:
            filename = rootpath + outpath + str(batch_id * batch_size + i) + '_predicted_' + labelname + labelnum + '.nii.gz'
        else:
            filename = rootpath + outpath + str(i) + '_predicted_' + labelname + '.nii.gz'
        print(filename)
        nibabel.save(new_img, filename)
