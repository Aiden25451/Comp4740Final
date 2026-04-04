import numpy as np
from skimage import io, color
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from skimage import io, color
from skimage.segmentation import mark_boundaries


def spectral_superpixels(
    image,
    n_segments=200,
    color_weight=1.0,
    spatial_weight=0.5,
    gamma_color=0.001,
    gamma_space=0.001,
):
    """
    Segment an image into superpixels using spectral clustering.

    Parameters
    ----------
    image : ndarray (H, W, 3)
        Input RGB image.
    n_segments : int
        Number of superpixels.
    color_weight : float
        Weight for color features.
    spatial_weight : float
        Weight for spatial features.
    gamma_color : float
        RBF gamma for color similarity.
    gamma_space : float
        RBF gamma for spatial similarity.

    Returns
    -------
    labels : ndarray (H, W)
        Superpixel labels.
    """

    H, W, _ = image.shape

    # Convert to Lab for perceptual uniformity
    lab = color.rgb2lab(image)

    # Build feature vectors
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    spatial = np.stack([yy, xx], axis=-1)

    # Flatten
    color_feat = lab.reshape(-1, 3)
    spatial_feat = spatial.reshape(-1, 2)

    # Normalize
    color_feat = color_feat / np.std(color_feat)
    spatial_feat = spatial_feat / np.std(spatial_feat)

    # Weighted concatenation
    X = np.hstack([color_weight * color_feat, spatial_weight * spatial_feat])

    # (This is the expensive step)
    # Build sparse kNN graph
    A = kneighbors_graph(X, n_neighbors=20, mode="distance", include_self=False)
    print("DONE building kNN graph")

    # Apply Gaussian kernel only to neighbor distances
    A.data = np.exp(-(A.data**2) * gamma_color)
    print("DONE applying Gaussian kernel")

    # Run spectral clustering using sparse nearest-neighbor affinity
    sc = SpectralClustering(
        n_clusters=n_segments,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
    )
    print("DONE spectral clustering")

    # IMPORTANT: fit_predict takes X, not A
    labels = sc.fit_predict(A)
    print("Fit and predict done")

    return labels.reshape(H, W)


# Execution for the function
# Execution for the function
image = io.imread("images/bird-water.jpg")
labels = spectral_superpixels(image, n_segments=50, spatial_weight=5.0)

# Create clean black boundaries
overlay = mark_boundaries(image, labels, color=(0, 0, 0))

# Save the result
io.imsave("output/superpixels.png", (overlay * 255).astype("uint8"))
