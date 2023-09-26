import numpy as np
import nibabel as nib

def normalize(path):
    nifti_image = nib.load(path)
    data = nifti_image.get_fdata()  # Get the data as a numpy array
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def save_as_nifti(image_data, output_path, original_nifti_path):
    if isinstance(image_data, nib.Nifti1Image):
        nib.save(image_data, output_path)
    else:
        original_nifti = nib.load(original_nifti_path)
        new_img = nib.Nifti1Image(image_data, original_nifti.affine)
        nib.save(new_img, output_path)

def main():
    test_subjs=['GE_04', 'GE_05', 'GE_06', 'GE_07', 'GE_08', 'GE_09', 'GE_10', 'GE_11', 'GE_12', 'GE_13', 'GE_14', 'GE_15', 'GE_16', 'GE_17', 'GE_18', 'GE_19', 'GE_20', 'GE_21', 'GE_22', 'GE_23', 'GE_26', 'GE_27', 'GE_29', 'GE_30', 'GE_31', 'GE_32', 'GE_33', 'GE_36']

    for subj in test_subjs:       
        # masked img
        orig_img = '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/tSNR/orig_correct/' + subj + '/masked_data.nii'
        normalized_img = normalize(orig_img)
        save_as_nifti(normalized_img, '/data/GE_MCI/Scripts/DenoiseASL/NicoleTest/GE_MBME_ASL_RS/tSNR/orig_correct/' + subj + '/normalized_masked_data.nii', orig_img)


if __name__ == "__main__":
    main()
