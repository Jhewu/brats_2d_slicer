"""
This file is to check the raw values of the .nii.gz
files in 2d slice using numpy. It can also be used
to extract RGB images of the scan for visualization
Needs to be in the same directory as the .nii files
"""
import nibabel as nib
import matplotlib.pyplot as plt
import cv2 as cv
import os
import numpy as np

MIN_SLICE = 30
MAX_SLICE = 120
SAVE_AS_RGB = False
SAVE_AS_NP = False

cwd = os.getcwd()
items = os.listdir()
#scan = items[0]
scan = "BraTS-PED-00105-000-seg.nii.gz"

# load the .nii.gz file
scan = nib.load(scan)

# convert raw data
data = scan.get_fdata()

# enable interactive mode
plt.ion()

# create a figure and axis
fig, ax = plt.subplots()

# create variable to hold the color var
colorbar = None

for slice in range(MIN_SLICE, MAX_SLICE): 
    print(f"\nThis is slice {slice}")
    
    slice_2d = data[:, :, slice]

    print(slice_2d.shape)
    print(slice_2d.dtype)

    # clear the previous plot
    ax.clear()

    # plot the image
    im = plt.imshow(slice_2d, cmap='gray')

    if colorbar: 
        colorbar.remove()

    colorbar = plt.colorbar(im, ax=ax) 
    ax.set_title(f"Image Slice {slice}")
    ax.axis('off')  # Hide the axis

    # optional: save the slice
    if SAVE_AS_RGB or SAVE_AS_NP: 
        # Normalize the image data to the range
        slice_2d = slice_2d - np.min(slice_2d)  # Shift the min to 0
        slice_2d = slice_2d / np.max(slice_2d)  # Scale the max to 1

        if SAVE_AS_RGB:
            slice_2d = (slice_2d * 255).astype(np.uint8)  # Scale to 255 and convert to uint8
            cv.imwrite(f"slice{slice}.jpg", slice_2d)
        elif SAVE_AS_NP: 
            np.save(f"slice{slice}.npy", slice_2d)

    # draw the updated plot
    plt.draw()
    plt.pause(1)

# disable interactive mode
plt.ioff()

# close the plot
plt.close()

# plot the 2D slice
plt.imshow(slice_2d.T, cmap="gray", origin="lower")
plt.colorbar()
plt.title("2D slice")
plt.show()
