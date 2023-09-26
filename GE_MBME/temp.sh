#!/bin/bash

subjects=('GE_04' 'GE_05' 'GE_06' 'GE_07' 'GE_08' 'GE_09' 'GE_10' 'GE_11' 'GE_12' 'GE_13' 'GE_14' 'GE_15' 'GE_16' 'GE_17' 'GE_18' 'GE_19' 'GE_20' 'GE_21' 'GE_22' 'GE_23' 'GE_26' 'GE_27' 'GE_29' 'GE_30' 'GE_31' 'GE_32' 'GE_33' 'GE_36')

base_path="/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS"
result_path="${base_path}"
tSNR_path="${base_path}/tSNR/orig_correct"

for subject in "${subjects[@]}"; do
    3dTstat -stdev -prefix ${tSNR_path}/${subject}/rm.noise.all ${tSNR_path}/${subject}/normalized_masked_data.nii
    3dTstat -mean -prefix ${tSNR_path}/${subject}/rm.signal.all.nii ${tSNR_path}/${subject}/normalized_masked_data.nii
    3dcalc -a ${tSNR_path}/${subject}/rm.signal.all.nii -b ${tSNR_path}/${subject}/rm.noise.all+tlrc.BRIK -expr 'a/b' -prefix ${tSNR_path}/${subject}/tsnr2.nii
done
