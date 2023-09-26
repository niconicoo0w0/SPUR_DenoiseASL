import nibabel as nib
from nilearn.image import resample_img
import numpy as np

# test subject: GE_05
# subj for train: SUBIDS
# SUBIDS=['GE_05','GE_04','GE_18','GE_10','GE_12','GE_30','GE_19']
SUBIDS=['GE_06', 'GE_07', 'GE_08', 'GE_09', 'GE_11', 'GE_13', 'GE_14', 'GE_15', 'GE_16', 'GE_17','GE_20', 'GE_21', 'GE_22', 'GE_23', 'GE_26', 'GE_27', 'GE_29', 'GE_31', 'GE_32', 'GE_33', 'GE_36']

def get_target_affine(original_affine, original_shape, target_shape):
    scale_factors = [orig_dim / target_dim for orig_dim, target_dim in zip(original_shape, target_shape)]
    target_affine = np.copy(original_affine)
    for i in range(3):
        target_affine[:3, i] *= scale_factors[i]
    return target_affine

def resample_single_volume(volume, original_affine, target_affine, target_shape):
    temp_img = nib.Nifti1Image(volume, original_affine)
    resampled_temp_img = resample_img(temp_img, target_affine=target_affine, target_shape=target_shape, interpolation='continuous')
    return resampled_temp_img.get_fdata()

def crop_nifti(input_path, output_path, desired_dim):
    img = nib.load(input_path)
    original_affine = img.affine
    target_affine = get_target_affine(original_affine, img.shape[:3], desired_dim)
    
    if len(img.shape) == 4 and img.shape[3] > 1:
        resampled_data_list = [resample_single_volume(img.dataobj[..., t], original_affine, target_affine, desired_dim) 
                               for t in range(img.shape[3])]
        resampled_data = np.stack(resampled_data_list, axis=3)
    else:
        resampled_data = resample_single_volume(img.get_fdata(), original_affine, target_affine, desired_dim)
    
    resampled_img = nib.Nifti1Image(resampled_data, target_affine)
    resampled_img.to_filename(output_path)

def main():
    for subj in SUBIDS:         
        input_nifti_path = "/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/" + subj + "_MBME_PCASL_RS_hp200_s4.nii.gz"
        output_nifti_path = "/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/" + subj + "_MBME_PCASL_RS_hp200_s4_rs35.nii.gz"
        desired_dim = (52, 62, 52)
        crop_nifti(input_nifti_path, output_nifti_path, desired_dim)
        
    for subj in SUBIDS:         
        input_nifti_path = "/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/" + subj + "_MBME_PCASL_RS_hp200_s4_mean.nii.gz"
        output_nifti_path = "/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/" + subj + "_MBME_PCASL_RS_hp200_s4_mean_rs35.nii.gz"
        desired_dim = (52, 62, 52)
        crop_nifti(input_nifti_path, output_nifti_path, desired_dim)

if __name__ == "__main__":
    main()
