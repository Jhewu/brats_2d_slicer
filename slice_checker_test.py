"""
Needs to be in the same directory as the .nii files
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

MIN_SLICE = 30
MAX_SLICE = 120

# enable interactive mode
plt.ion()

# create a figure and axis
fig, ax = plt.subplots()

# create variable to hold the color var
colorbar = None

for slice in range(MIN_SLICE, MAX_SLICE): 
    print(f"\nThis is slice {slice}")

    image = np.load(f"slice{slice}.npy")

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