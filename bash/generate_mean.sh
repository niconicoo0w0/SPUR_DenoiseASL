#!/bin/bash

FOLDER="/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/ASL/RS"

FILES=("GE_04_GE_MBME_RS_ASL_s4.nii.gz")
# "GE_05_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_06_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_18R_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_19_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_19R_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_06R_GE_MBME_RS_ASL_s4.nii.gz"  
# "GE_20_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_07_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_21_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_07R_GE_MBME_RS_ASL_s4.nii.gz"  
# "GE_21R_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_08_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_22_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_08R_GE_MBME_RS_ASL_s4.nii.gz"  
# "GE_22R_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_09_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_23_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_09R_GE_MBME_RS_ASL_s4.nii.gz"  
# "GE_26_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_10_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_27_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_10R_GE_MBME_RS_ASL_s4.nii.gz"  
# "GE_29_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_11_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_29R_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_12_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_30_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_12R_GE_MBME_RS_ASL_s4.nii.gz"  
# "GE_30R_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_13_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_31_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_13R_GE_MBME_RS_ASL_s4.nii.gz"  
# "GE_31R_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_14_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_32_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_15_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_32R_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_15R_GE_MBME_RS_ASL_s4.nii.gz"  
# "GE_33_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_16_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_33R_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_17_GE_MBME_RS_ASL_s4.nii.gz"   
# "GE_36_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_17R_GE_MBME_RS_ASL_s4.nii.gz"  
# "GE_18_GE_MBME_RS_ASL_s4.nii.gz"
# "GE_36R_GE_MBME_RS_ASL_s4.nii.gz")


for file in "${FILES[@]}"; do
    input_path="${FOLDER}/${file}"
    output_path="${FOLDER}/$(basename $file .nii.gz)_mean.nii.gz"

    fslmaths $input_path -Tmean $output_path
    
    echo "Processed: $file"
done

echo "All files processed!"
