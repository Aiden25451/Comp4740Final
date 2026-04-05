import numpy as np
from pyarrow import dataset
from pyarrow import dataset
from skimage.graph import rag_mean_color
from process_dataset import tokenize
from datasets import load_dataset
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import networkx as nx


def superpixel_features(img, segments):
    """
    Compute per-superpixel features (mean RGB colour + centroid x/y).

    Parameters
    ----------
    img      : np.ndarray, shape (H, W, 3), uint8
    segments : np.ndarray, shape (H, W), int  – SLIC output

    Returns
    -------
    features : np.ndarray, shape (num_superpixels, 5)
                columns: [mean_r, mean_g, mean_b, centre_x, centre_y]
                All values are normalised to [0, 1].
    """
    h, w = segments.shape
    num_sp = segments.max() + 1

    features = np.zeros((num_sp, 5), dtype=np.float32)

    for sp_id in range(num_sp):
        mask = segments == sp_id
        if mask.sum() == 0:
            continue
        # mean colour
        mean_rgb = img[mask].mean(axis=0) / 255.0  # shape (3,)
        # centroid
        ys, xs = np.where(mask)
        cx = xs.mean() / w
        cy = ys.mean() / h

        features[sp_id] = [mean_rgb[0], mean_rgb[1], mean_rgb[2], cx, cy]

    return features


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


def graph_convert_example():
    # Example usage
    dataset = load_dataset("Simezu/brain-tumour-MRI-scan", split="train")
    dataset = dataset.train_test_split(test_size=10, train_size=10, seed=1)
    dataset = dataset.map(tokenize, batched=True, batch_size=64, num_proc=16)

    # Save images
    first_img = np.array(dataset["train"][0]["image"].convert("RGB"))
    first_seg = np.array(dataset["train"][0]["slic"])

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
    axes[1].set_title(f"SLIC Superpixels ({first_seg.max() + 1})")
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
