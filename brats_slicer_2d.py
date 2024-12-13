"""
This script takes in a BraTS
dataset with two directories, a 
training directory and a validation 
directory and then create two new directory 
train_data and val_data with the respective
mpMRI in slices

You can modify where the slicing occurs
with MIN_SLICE and MAX_SLICE (z-coordinates)
in axial view of the brain

By default it saves the 2d slices as .npy files 
so that there's no need to normalize and making
obtainig ground truth masks easier. You can modify that
by changing the save_as_np to False in the function GetImageSlices()
"""

"""Imports"""
import os
import nibabel as nib
import cv2 as cv
import numpy as np

"""HYPERPARAMETERS"""
TRAINING_FOLDER = "training"
VALIDATION_FOLDER = "validation"

MIN_SLICE = 30
MAX_SLICE = 120

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def GetImageSlices(scan, scan_dir_dest, save_as_np=True, is_ground_truth=False): 
    print(f"Slicing at...{scan}")

    # convert raw data
    scan = nib.load(scan)
    scan = scan.get_fdata()
    
    # save slices from MIN_SLICE and MAX_SLICE
    for slice in range(MIN_SLICE, MAX_SLICE):
        # get 2d slice 
        slice_2d = scan[:, :, slice]

        # normalize the 2d slice to the range of [0, 1]
        slice_2d_min = np.min(slice_2d) 
        slice_2d_max = np.max(slice_2d)
        
        if is_ground_truth == False:
        # for ground truth, the default range is [0-3]. We
        # will not normalize it because then we can easily 
        # extract the ground truth mask, if you still want to 
        # normalize it, you can disable it by turning is_ground_true = False
            if slice_2d_max > 0:
                # Normalize the slice
                slice_2d = (slice_2d - slice_2d_min) / slice_2d_max 
                    #slice_2d = slice_2d - np.min(slice_2d)  
                    #slice_2d = slice_2d / np.max(slice_2d) 
            else: 
                print(f"Warning: Maximum value in slice {slice} is zero. Normalization skipped.") 
                slice_2d = slice_2d - slice_2d_min # At least shift the min to 0
        
        # save the slice 2d into the respective directory
        slice_name = os.path.join(scan_dir_dest, f"slice{slice}")
        if save_as_np: 
            np.save(slice_name, slice_2d)
        else: 
            cv.imwrite(slice_name, slice_2d)

"""Main Runtime"""
def BraTS_Slicer_2D(): 
    # set up cwd and training and validation paths
    root_dir = os.getcwd()
    training_dir = os.path.join(root_dir, TRAINING_FOLDER)
    validation_dir = os.path.join(root_dir, VALIDATION_FOLDER)

    # list of directories in training and validation
    patients_train_list = os.listdir(training_dir)
    patients_val_list = os.listdir(validation_dir)

    # create the train_data and val_data destination directory 
    train_data_dest = "train_data"
    val_data_dest = "val_data"
        #CreateDir(train_data_dest)
        #CreateDir(val_data_dest)

    # perform slicing on training dataset
    for patient in patients_train_list: 
        # create the destination directory for each patient
        patient_dir_dest = f"{train_data_dest}/{patient}"
            #CreateDir(patient_dir_dest)

        # list of directories in each patient
        patient_train_dir = os.path.join(training_dir, patient)
        patient_train_scan_list = os.listdir(patient_train_dir)

        # slicing 2d images for each patient scan and saving it to train_data
        for scan in patient_train_scan_list: 
            ground_truth = f"{patient}-seg.nii.gz"

            # create the path where the scan is located
            scan_path = os.path.join(patient_train_dir, scan)

            # create destination directory for each scan modality
            scan_dir_name = scan.replace(".nii.gz", "")
            scan_dir_dest = f"{patient_dir_dest}/{scan_dir_name}"
            CreateDir(scan_dir_dest)

            # separate the ground truth
            if scan == ground_truth: 
                GetImageSlices(scan_path, scan_dir_dest, save_as_np=True, is_ground_truth=True)
            else: 
                GetImageSlices(scan_path, scan_dir_dest, save_as_np=True, is_ground_truth=False)

    # perform slicing on validation dataset
    for patient in patients_val_list: 
        # create the destination directory for each patient
        patient_dir_dest = f"{val_data_dest}/{patient}"
            #CreateDir(patient_dir_dest)

        # list of directories in each patient
        patient_val_dir = os.path.join(validation_dir, patient)
        patient_val_scan_list = os.listdir(patient_val_dir)

        # slicing 2d images and saving from validation set
        for scan in patient_val_scan_list: 
            # create the path where the scan is located
            scan_path = os.path.join(patient_val_dir, scan)

            # create destination directory for each scan modality
            scan_dir_name = scan.replace(".nii.gz", "")
            scan_dir_dest = f"{patient_dir_dest}/{scan_dir_name}"
            CreateDir(scan_dir_dest)

            # get 2d slices and save to scan_dir_dest
            GetImageSlices(scan_path, scan_dir_dest, save_as_np=True, is_ground_truth=False)

if __name__ == "__main__": 
    print(f"Creating slices from {MIN_SLICE} to {MAX_SLICE} (representing the z-coordinates) in axial view...\n")
    BraTS_Slicer_2D()
    print("\nFinish slicing, please check your directory for train_data and val_data\n")
