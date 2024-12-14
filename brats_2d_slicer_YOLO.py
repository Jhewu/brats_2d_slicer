"""
This script is an alternative of brats_2d_slicer
specifically made to prepare data for Yolov8-seg 
training. It creates two directories, a train_data
and a gt_data. It's recommended that you use a separate
script to split the training data into test, val and train 
for your actual training

You can modify where the slicing occurs
with MIN_SLICE and MAX_SLICE (z-coordinates)
in axial view of the brain

By default it saves the 2d slices as .npy files 
so that there's no need to normalize. You can modify that
by changing the save_as_np to False in the function GetImageSlices()

Only the training 2d slices are normalized, the ground_truth 
is kept the default 0-3 range for easier mask extraction during training. 
If you still want to normalize it, you can disable it by turning is_ground_true = False
"""

"""Imports"""
import os
import nibabel as nib
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

"""HYPERPARAMETERS"""
TRAINING_FOLDER = "training"
VALIDATION_FOLDER = "validation"

MIN_SLICE = 30
MAX_SLICE = 120

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def GetImageSlices(scan_name, scan_path, scan_dir_dest, save_as_np=True, is_ground_truth=False): 
    print(f"Slicing at...{scan_path}")

    # convert raw data
    scan = nib.load(scan_path)
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
        slice_name = os.path.join(scan_dir_dest, f"{scan_name}{slice}")
        if save_as_np: 
            np.save(slice_name, slice_2d)
        else: 
            cv.imwrite(slice_name, slice_2d)

def GetPatientScan(patient, patient_data_dest, gt_data_dest, patient_source_dir): 
    # list of directories in each patient
    patient_data_dir = os.path.join(patient_source_dir, patient)
    patient_data_scan_list = os.listdir(patient_data_dir)

    # slicing 2d images for each patient scan and saving it to train_data
    for scan in patient_data_scan_list: 
        ground_truth = f"{patient}-seg.nii.gz"

        # create the path where the scan is located
        scan_path = os.path.join(patient_data_dir, scan)

        # create the destination directories
        CreateDir(patient_data_dest)
        CreateDir(gt_data_dest)

        # create the scan directory name
        scan_dir_name = scan.replace(".nii.gz", "")

        # separate the ground truth
        if scan == ground_truth: 
            GetImageSlices(scan_dir_name, scan_path, gt_data_dest, save_as_np=True, is_ground_truth=True)
        else: 
            GetImageSlices(scan_dir_name, scan_path, patient_data_dest, save_as_np=True, is_ground_truth=False)

"""Main Runtime"""
def BraTS_2D_Slicer_YOLO(): 
    # set up cwd and training and validation paths
    root_dir = os.getcwd()
    training_dir = os.path.join(root_dir, TRAINING_FOLDER)

    # list of directories in training and validation
    patients_train_list = os.listdir(training_dir)

    # create the train_data and gt_data destination directory 
    train_data_dest = "train_data"
    gt_data_dest = "gt_data"

    # define a thread pool executor with a maximum numnber of workers
    max_workers = 10 # adjust based on your syster's capabilities

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # perform slicing on training dataset
        for patient in patients_train_list: 
            executor.submit(GetPatientScan, patient, train_data_dest, gt_data_dest, training_dir)

if __name__ == "__main__": 
    print(f"Creating slices from {MIN_SLICE} to {MAX_SLICE} (representing the z-coordinates) in axial view...\n")
    BraTS_2D_Slicer_YOLO()
    print("\nFinish slicing, please check your directory for train_data\n")
