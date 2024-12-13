"""
Needs to be in the same directory as the .nii files
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = np.load("slice90.npy")

print(image.shape)
print(image.dtype)

# Plot the image
plt.imshow(image, cmap='gray')
plt.colorbar()  # Optional: Add a color bar
plt.title("Image Plot")
plt.axis('off')  # Hide the axis
plt.show()
