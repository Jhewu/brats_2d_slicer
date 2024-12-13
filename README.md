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