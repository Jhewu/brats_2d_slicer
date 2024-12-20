This script takes in a BraTS dataset with two directories, a 
training directory and a validation directory and then create
two new directory train_data and val_data with the respective
mpMRI in slices

You can modify where the slicing occurs with MIN_SLICE and MAX_SLICE
(z-coordinates) in axial view of the brain

By default it saves the 2d slices as .npy files 
so that there's no need to normalize. You can modify that
by changing the save_as_np to False in the function GetImageSlices()

Only the training 2d slices are normalized, the ground_truth 
is kept the default 0-3 range for easier mask extraction during training. 
If you still want to normalize it, you can disable it by turning is_ground_true = False

Brats_2d_slicer_YOLO on the otherhand is an alternative of brats_2d_slicer
specifically made to prepare data for Yolov8-seg 
training. It creates two directories, a train_data
and a gt_data. It's recommended that you use a separate
script to split the training data into test, val and train 
for your actual training

Since YOLOv8-seg training data and mask needs to contain
the same name, in this script, it only picks one modality. 
Although you could change it depending on which modality you want
or run it multiple times and then stitch them together at the end. 

There are also two accompanying tools/mini scripts: 

slice_checker_raw is used to check the raw values of the .nii.gz files in 2d slice
using numpy. It can also be used to extract RGB images of the scan for visualization. 
It plots throught each slicesequentially.

slice_checker_after, checks and visualize the 2d slices after being run through 
brats_slicer_2d.py. It plots throught each slice sequentially. 

Both of these files slice_checker_raw and slice_checker_after needs to be in the 
same directory as the 2d_slices in .npy array. If you want to check for 
RGB just change the extension to .JPG
