#!/bin/bash

# subjects=('GE_04' 'GE_05' 'GE_06' 'GE_07' 'GE_08' 'GE_09' 'GE_10' 'GE_11' 'GE_12' 'GE_13' 'GE_14' 'GE_15' 'GE_16' 'GE_17' 'GE_18' 'GE_19' 'GE_20' 'GE_21' 'GE_22' 'GE_23' 'GE_26' 'GE_27' 'GE_29' 'GE_30' 'GE_31' 'GE_32' 'GE_33' 'GE_36')
subjects=('GE_04' 'GE_05' 'GE_06' 'GE_07' 'GE_08' 'GE_09' 'GE_10' 'GE_11' 'GE_12' 'GE_13' 'GE_14' 'GE_15' 'GE_16' 'GE_17' 'GE_18')

base_path="/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS"
result_path="${base_path}/Result"
tSNR_path="${base_path}/tSNR/lstm"

for subject in "${subjects[@]}"; do
    mkdir "${tSNR_path}/${subject}_denoised"    
    3dTstat -stdev -prefix ${tSNR_path}/${subject}_denoised/rm.noise.all ${result_path}/${subject}_denoised.nii.gz
    3dAutomask -prefix ${tSNR_path}/${subject}_denoised/full_mask.nii ${result_path}/${subject}_denoised.nii.gz
    3dTstat -mean -prefix ${tSNR_path}/${subject}_denoised/rm.signal.all.nii ${result_path}/${subject}_denoised.nii.gz
    3dcalc -a ${tSNR_path}/${subject}_denoised/rm.signal.all.nii -b ${tSNR_path}/${subject}_denoised/rm.noise.all+tlrc.BRIK -c ${tSNR_path}/${subject}_denoised/full_mask.nii -expr 'c*a/b' -prefix ${tSNR_path}/${subject}_denoised/tSNR_after.nii
done
