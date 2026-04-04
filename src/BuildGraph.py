import numpy as np
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph


def superpixels_to_graph(img, segments, k=8):
    """
    Convert SLIC superpixels into a graph suitable for GNNs.

    Returns:
        x: node features (num_superpixels, feature_dim)
        edge_index: graph edges (2, num_edges)
    """
    H, W = segments.shape
    num_segments = segments.max() + 1

    # ---------------------------------------------------------
    # 1. Compute region features
    # ---------------------------------------------------------
    props = regionprops(segments + 1, intensity_image=img.mean(axis=-1))

    # Example features: mean RGB + centroid
    feats = []
    centroids = []

    for sp in range(num_segments):
        mask = segments == sp
        region = props[sp]

        mean_color = img[mask].mean(axis=0)  # (3,)
        centroid = np.array(region.centroid)  # (2,)

        feats.append(np.concatenate([mean_color, centroid]))
        centroids.append(centroid)

    x = np.vstack(feats)  # shape (N, 5)

    # ---------------------------------------------------------
    # 2. Build adjacency using k-NN on centroids
    # ---------------------------------------------------------
    centroids = np.vstack(centroids)
    A = kneighbors_graph(centroids, k, mode="connectivity", include_self=False)
    A = A.toarray()

    # Convert adjacency to edge_index
    edge_index = np.vstack(np.nonzero(A))

    return x.astype(np.float32), edge_index.astype(np.int64)
