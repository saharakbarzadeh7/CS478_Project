import numpy as np
import nibabel as nib
#import pandas as pd
import os


def compute_weighted_combination(t1, t2, t1ce, flair):
    """
    Compute the weighted combination of MRI modalities.

    Parameters:
        t1 (np.ndarray): 3D array for T1 modality.
        t2 (np.ndarray): 3D array for T2 modality.
        t1ce (np.ndarray): 3D array for T1CE modality.
        flair (np.ndarray): 3D array for FLAIR modality.

    Returns:
        np.ndarray: 3D array of the weighted combination.
    """
    # Ensure all modalities have the same shape
    if not (t1.shape == t2.shape == t1ce.shape == flair.shape):
        raise ValueError("All input modalities must have the same shape.")

    # Compute the weighted combination
    weighted_combination = (t1 + 2 * t2 + 3 * t1ce + 3 * flair) / 9
    return weighted_combination


rootdir = 'D:/CSUSM-Computer Science/Deep Learning/starter_code/starter_code'
subdirs = os.listdir(rootdir)

for sub in subdirs:
    t1_file_path = rootdir + '/' + sub + '/' + sub + '_t1.nii.gz'
    t1ce_file_path = rootdir + '/' + sub + '/' + sub + '_t1ce.nii.gz'
    t2_file_path = rootdir + '/' + sub + '/' + sub + '_t2.nii.gz'
    flair_file_path = rootdir + '/' + sub + '/' + sub + '_flair.nii.gz'

    if os.path.exists(t1_file_path):
        nii_img = nib.load(t1_file_path)
        t1 = nii_img.get_fdata()
        nii_img = nib.load(t1ce_file_path)
        t1ce = nii_img.get_fdata()
        nii_img = nib.load(t2_file_path)
        t2 = nii_img.get_fdata()
        nii_img = nib.load(flair_file_path)
        flair = nii_img.get_fdata()

        weighted_image = compute_weighted_combination(t1, t2, t1ce, flair)
        img = nib.Nifti1Image(weighted_image, nii_img.affine, nii_img.header)
        save_path = rootdir + '/' + sub + '/' + sub + '_weighted1233.nii.gz'
        print(save_path)
        nib.save(img, save_path)


