"""
Needs to be in the same directory as the .nii files
"""

import nibabel as nib
import matplotlib.pyplot as plt
import cv2 as cv
import os
import numpy as np

cwd = os.getcwd()
items = os.listdir()
dir = os.path.join(cwd, items[0])

# load the .nii.gz file
print(items[0])
img = nib.load(items[0])

# get the data as a Numpy array
data = img.get_fdata()
print(data.dtype)
print(data.shape)

# select a 2D slice (e.g. the middle slice along the z-axis)
slice_index = data.shape[2] // 2
slice_2d = data[:, :, 30]

# Normalize the image data to the range [0, 255]
slice_2d = slice_2d - np.min(slice_2d)  # Shift the min to 0
slice_2d = slice_2d / np.max(slice_2d)  # Scale the max to 1
slice_2d = (slice_2d * 255).astype(np.uint8)  # Scale to 255 and convert to uint8

print(slice_2d.shape)
print(slice_2d.dtype)

#cv.imwrite("ground_truth.jpg", slice_2d)
#np.save("image.npy", slice_2d)

# plot the 2D slice
plt.imshow(slice_2d.T, cmap="gray", origin="lower")
plt.colorbar()
plt.title("2D slice")
plt.show()
