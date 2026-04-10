from matplotlib.pyplot import gray
import numpy as np
from pyarrow import dataset
from pyarrow import dataset
from skimage.graph import rag_mean_color
from process_dataset import tokenize
from datasets import load_dataset
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import regionprops


def compute_geometry_features(segments, num_sp):
    # regionprops expects labels starting at 1
    props = regionprops(segments + 1)

    geom = np.zeros((num_sp, 2), dtype=np.float32)

    for sp in range(num_sp):
        p = props[sp]

        area = p.area
        solidity = p.solidity

        geom[sp] = [
            area,
            solidity,
        ]

    return geom


def compute_glcm_features(gray, segments, num_sp):
    # 4 orientations used in the paper
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm_feats = np.zeros((num_sp, 4), dtype=np.float32)
    # contrast, homogeneity, energy, entropy

    for sp in range(num_sp):
        mask = segments == sp
        patch = gray[mask].astype(np.uint8)

        if len(patch) < 5:
            continue

        # Build GLCM for all 4 angles
        glcm = graycomatrix(
            patch.reshape(-1, 1),
            distances=[1],
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True,
        )

        # Average Haralick features across angles
        contrast = graycoprops(glcm, "contrast").mean()
        homogeneity = graycoprops(glcm, "homogeneity").mean()
        energy = graycoprops(glcm, "energy").mean()

        # Entropy must be computed manually
        entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

        glcm_feats[sp] = [contrast, homogeneity, energy, entropy]

    return glcm_feats


def compute_lbp_hist(gray, segments, num_sp, P=8, R=1, bins=10):
    # Ensure grayscale is 2D uint8
    gray = np.squeeze(gray).astype(np.uint8)

    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    hist = np.zeros((num_sp, bins), dtype=np.float32)

    for sp in range(num_sp):
        mask = segments == sp
        vals = lbp[mask].astype(int)
        h, _ = np.histogram(vals, bins=bins, range=(0, bins), density=True)
        hist[sp] = h

    return hist


def compute_color_and_centroids(img, segments, num_sp):
    """Compute mean RGB + normalized centroid per superpixel."""
    h, w = segments.shape

    flat_seg = segments.ravel()
    flat_img = img.reshape(-1, 3).astype(np.float32) / 255.0

    rgb_sum = np.zeros((num_sp, 3), dtype=np.float32)
    counts = np.zeros(num_sp, dtype=np.float32)

    np.add.at(rgb_sum, flat_seg, flat_img)
    np.add.at(counts, flat_seg, 1)

    mean_rgb = rgb_sum / counts[:, None]

    ys, xs = np.mgrid[0:h, 0:w]
    cx_sum = np.zeros(num_sp, dtype=np.float32)
    cy_sum = np.zeros(num_sp, dtype=np.float32)

    np.add.at(cx_sum, flat_seg, xs.ravel())
    np.add.at(cy_sum, flat_seg, ys.ravel())

    cx = cx_sum / counts / w
    cy = cy_sum / counts / h

    return mean_rgb, cx, cy


def superpixel_features(img, segments):
    """Return a clean feature matrix for all superpixels."""
    num_sp = segments.max() + 1

    # Basic grayscale
    gray = img.mean(axis=2).astype(np.uint8)

    # Modular feature calls
    mean_rgb, cx, cy = compute_color_and_centroids(img, segments, num_sp)
    lbp_hist = compute_lbp_hist(gray, segments, num_sp)
    # glcm_feats = compute_glcm_features(gray, segments, num_sp)
    # geom_feats = compute_geometry_features(segments, num_sp)

    # Stack everything
    return np.column_stack([mean_rgb, cx, cy, lbp_hist])


def superpixels_to_graph(img, segments):
    """
    Build a graph from a SLIC segmentation map.

    Two superpixels are connected by an edge when they share at least one
    pixel boundary (RAG – Region Adjacency Graph).

    Parameters
    ----------
    img      : np.ndarray, shape (H, W, 3), uint8
    segments : np.ndarray, shape (H, W), int

    Returns
    -------
    node_features : np.ndarray, shape (num_superpixels, 5)
    edge_index    : np.ndarray, shape (2, num_edges)   – COO format, undirected
    """
    # --- node features ---
    node_features = superpixel_features(img, segments)

    # --- edges via scikit-image RAG ---
    rag = rag_mean_color(img, segments)

    src_nodes = []
    dst_nodes = []
    for u, v in rag.edges():
        # add both directions so the graph is undirected
        src_nodes.append(u)
        dst_nodes.append(v)
        src_nodes.append(v)
        dst_nodes.append(u)

    edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)  # (2, E)

    return node_features, edge_index


def graph_to_data(node_features, edge_index, label):
    """
    Convert numpy graph arrays into a PyTorch Geometric Data object.

    Parameters
    ----------
    node_features : np.ndarray, shape (num_nodes, 5)
    edge_index    : np.ndarray, shape (2, num_edges)
    label         : int  – class index (e.g. 0 or 1)

    Returns
    -------
    torch_geometric.data.Data
    """
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def tokenize_graph(batch):
    """Apply graph conversion to a batch of images with SLIC segments."""
    node_features_list = []
    edge_index_list = []
    for pil_img, seg in zip(batch["image"], batch["superpixels"]):
        img_np = np.array(pil_img.convert("RGB"))
        seg_np = np.array(seg)
        nf, ei = superpixels_to_graph(img_np, seg_np)
        node_features_list.append(nf)
        edge_index_list.append(ei)
    return {"node_features": node_features_list, "edge_index": edge_index_list}


def graph_convert_example():
    # Example usage
    dataset = load_dataset("Simezu/brain-tumour-MRI-scan", split="train")
    dataset = dataset.train_test_split(test_size=10, train_size=10, seed=1)
    dataset = dataset.map(
        tokenize, batched=True, batch_size=64, num_proc=16, load_from_cache_file=False
    )

    # Save images
    first_img = np.array(dataset["train"][0]["image"].convert("RGB"))
    first_seg = np.array(dataset["train"][0]["superpixels"])

    node_features, edge_index = superpixels_to_graph(first_img, first_seg)
    print("Node features shape:", node_features.shape)  # (num_superpixels, 5)
    print("Edge index shape:", edge_index.shape)  # (2, num_edges)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Image → Superpixels → Graph", fontsize=14, fontweight="bold")

    # Panel 1 – original image
    axes[0].imshow(first_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Panel 2 – SLIC boundaries
    axes[1].imshow(mark_boundaries(first_img, first_seg, color=(1, 1, 0)))
    axes[1].set_title(f"Superpixels ({first_seg.max() + 1})")
    axes[1].axis("off")

    # Panel 3 – graph overlaid on image
    G = nx.Graph()
    G.add_nodes_from(range(len(node_features)))
    for u, v in zip(edge_index[0], edge_index[1]):
        if u < v:
            G.add_edge(int(u), int(v))

    h, w = first_seg.shape
    pos = {
        i: (node_features[i, 3] * w, node_features[i, 4] * h)
        for i in range(len(node_features))
    }
    node_colors = node_features[:, :3].clip(0, 1).tolist()

    axes[2].imshow(first_img, alpha=0.4)
    nx.draw(
        G,
        pos=pos,
        ax=axes[2],
        node_size=15,
        node_color=node_colors,
        edge_color="white",
        width=0.4,
        with_labels=False,
    )
    axes[2].set_title(
        f"RAG Graph ({len(node_features)} nodes, {edge_index.shape[1]//2} edges)"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("output/graph_viz.png", dpi=150, bbox_inches="tight")
    print("Saved → output/graph_viz.png")


graph_convert_example()
