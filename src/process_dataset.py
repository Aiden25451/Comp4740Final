from datasets import load_dataset
from skimage.segmentation import slic
import numpy as np
import matplotlib.pyplot as plt


def run_slic(img, n_segments=200, compactness=10):
    """
    Runs SLIC superpixel segmentation.
    Returns a segmentation map of shape (H, W).
    """
    segments = slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
    )
    return segments


# Preparing the dataset
def tokenize(batch):
    """Apply SLIC to a batch of images."""
    imgs = batch["image"]
    results = []

    for pil_img in imgs:
        img_np = np.array(pil_img.convert("RGB"))
        seg = run_slic(img_np)
        results.append(seg)

    # return something so HF is satisfied
    return {"slic": results}


def preprocess_dataset_example():
    dataset = load_dataset("Simezu/brain-tumour-MRI-scan", split="train")
    dataset = dataset.train_test_split(test_size=100, train_size=2000, seed=1)
    dataset = dataset.map(tokenize, batched=True, batch_size=64, num_proc=16)

    # Save images
    first_img = np.array(dataset["train"][0]["image"])
    first_seg = dataset["train"][0]["slic"]
    plt.imsave("output/original.png", first_img)
    plt.imsave("output/slic.png", first_seg)
