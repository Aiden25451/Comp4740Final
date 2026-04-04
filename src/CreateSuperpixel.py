import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float


# ---------------------------------------------------------
# 1. Load image and convert to correct format
# ---------------------------------------------------------
def load_image(path):
    """
    Loads an image and converts it to float [0,1] format.
    Returns an array of shape (H, W, 3).
    """
    img = imread(path)
    img = img_as_float(img)  # ensures float64 in [0,1]
    if img.ndim == 2:  # grayscale → RGB
        img = np.stack([img] * 3, axis=-1)
    return img


# ---------------------------------------------------------
# 2. Run SLIC segmentation
# ---------------------------------------------------------
def run_slic(img, n_segments=200, compactness=10):
    """
    Runs SLIC superpixel segmentation.
    Returns a segmentation map of shape (H, W).
    """
    segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)
    return segments


# ---------------------------------------------------------
# 3. Run LSC segmentation
# ---------------------------------------------------------
# def run_lsc(img, n_segments=200, compactness=0.1):
#     """
#     Runs LSC (Linear Spectral Clustering) segmentation.
#     Returns a segmentation map of shape (H, W).
#     """
#     segments = lsc(img, n_segments=n_segments, compactness=compactness, start_label=0)
#     return segments


# ---------------------------------------------------------
# 4. Overlay segmentation boundaries on original image
# ---------------------------------------------------------
def draw_boundaries(img, segments):
    """
    Returns an image with black boundaries drawn around superpixels.
    """
    return mark_boundaries(img, segments, color=(0, 0, 0))


# ---------------------------------------------------------
# Demo usage
# ---------------------------------------------------------
os.makedirs("output", exist_ok=True)

# Load example image
img = load_image("images/bird-water.jpg")  # <-- replace with your image path

# Run segmentations
seg_slic = run_slic(img, n_segments=50, compactness=50)
# seg_lsc = run_lsc(img, n_segments=200)

# Draw boundaries
slic_vis = draw_boundaries(img, seg_slic)
# lsc_vis = draw_boundaries(img, seg_lsc)

# Save results
plt.imsave("output/slic_result.png", slic_vis)
# plt.imsave("output/lsc_result.png", lsc_vis)

print("Saved output/slic_result.png and output/lsc_result.png")
