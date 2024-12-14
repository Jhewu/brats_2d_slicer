"""
Checks and visualize the 2d slices after being run
through brats_slicer_2d.py. Plots throught each slice
sequentially. Needs to be in the same directory as 
the 2d_slices in .npy array. If you want to check for 
RGB just change the extension to .JPG
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import os

MIN_SLICE = 30
MAX_SLICE = 120

# enable interactive mode
plt.ion()

# create a figure and axis
fig, ax = plt.subplots()

# create variable to hold the color var
colorbar = None

# get working directory
cwd = os.getcwd()

for slice in range(MIN_SLICE, MAX_SLICE): 
    print(f"\nThis is slice {slice}")
    #print(f"{cwd}{slice}.npy")

    image = np.load(f"{os.path.basename(cwd)}{slice}.npy")

    print(image.shape)
    print(image.dtype)

    # clear the previous plot
    ax.clear()

    # plot the image
    im = plt.imshow(image, cmap='gray')

    if colorbar: 
        colorbar.remove()

    colorbar = plt.colorbar(im, ax=ax) 
    ax.set_title(f"Image Slice {slice}")
    ax.axis('off')  # Hide the axis

    # draw the updated plot
    plt.draw()
    plt.pause(1)

# disable interactive mode
plt.ioff()

# close the plot
plt.close(fig)