# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zLUxkLSTYveQhRfrBLOrFE8-YH66APqN
"""

import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, models

def WideActivationResidualBlock(x, filters, dilation_rate=1):
    res = x
    x = layers.Conv3D(filters*4, (3,3,3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv3D(filters, (3,3,3), dilation_rate=dilation_rate, padding='same')(x)
    return layers.Add()([res, x])

def DWAN(input_shape=(64, 64, 21, 1)):
    input_img = layers.Input(shape=input_shape)

    x = layers.Conv3D(32, (3,3,3), padding='same')(input_img)
    res = x

    # Local Pathway
    local_path = x
    for _ in range(4):
        local_path = WideActivationResidualBlock(local_path, 32)

    # Global Pathway
    global_path = x
    for i in range(4):
        global_path = WideActivationResidualBlock(global_path, 32, dilation_rate=[2**i, 2**i, 2**i])

    # Concatenating the outputs of local and global pathways
    x = layers.Concatenate(axis=-1)([local_path, global_path])

    # Final layer
    output_img = layers.Conv3D(1, (3,3,3), padding='same')(x)

    # Adding the residual from the input to output
    output_img = layers.Add()([output_img, res])

    return models.Model(input_img, output_img)

# Load the .nii.gz file
nii_file = nib.load('/filtered_func_data_64_64_21_180.nii.gz')
data = nii_file.get_fdata()
data = np.transpose(data, (3, 0, 1, 2))  # Reordering dimensions to make time points as batch dimension
data = np.expand_dims(data, axis=-1)  # Adding channel dimension

# Create and compile the model
model = DWAN()
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model to your data
model.fit(data, data, epochs=10)  # This is an autoencoder, so input is the same as output

# Getting the output
output = model.predict(data)

# Save the output to .nii.gz file
output = np.squeeze(output)  # Removing channel dimension
output = np.transpose(output, (1, 2, 3, 0))  # Reordering dimensions back to original format
output_nii = nib.Nifti1Image(output, nii_file.affine)  # Using the same affine transformation as the original data
nib.save(output_nii, '/output_data_64_64_21_180.nii.gz')