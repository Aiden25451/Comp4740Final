from datasets import load_dataset
from skimage.segmentation import slic, felzenszwalb, quickshift
import numpy as np
import matplotlib.pyplot as plt
from config import config
import cv2

def run_slic(img, compactness=10):
    return slic(
        img,
        n_segments=config["segments"],
        compactness=compactness,
        start_label=0,
    )


def run_felzenszwalb(img, scale=100):
    return felzenszwalb(img, scale=scale)


def run_quickshift(img, kernel_size=3):
    return quickshift(img, kernel_size=kernel_size)


def run_lsc(img):
    """
    OpenCV LSC superpixels.
    Uses approximate region_size derived from desired segment count.
    """
    h, w = img.shape[:2]

    # Approximate region size from desired number of segments
    # region_size is closer to a linear size, not directly the number of segments
    region_size = max(5, int(np.sqrt((h * w) / config["segments"])))

    # OpenCV expects BGR, and docs recommend blur + Lab conversion
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    lsc = cv2.ximgproc.createSuperpixelLSC(
        img_lab,
        region_size=region_size,
        ratio=float(config["lsc_ratio"]),
    )
    lsc.iterate(int(config["lsc_iterations"]))

    labels = lsc.getLabels()
    return labels.astype(np.int32)

def run_superpixel(img):
    method = config["superpixel_method"]

    if method == "slic":
        return run_slic(img)
    elif method == "felzenszwalb":
        return run_felzenszwalb(img)
    elif method == "quickshift":
        return run_quickshift(img)
    elif method == "lsc":
        return run_lsc(img)
    else:
        raise ValueError(f"Unknown superpixel method: {method}")


def tokenize(batch):
    imgs = batch["image"]
    results = []

    for pil_img in imgs:
        img_np = np.array(pil_img.convert("RGB"))
        seg = run_superpixel(img_np)
        results.append(seg)

    return {"superpixels": results}

def save_superpixel_example():
    dataset = load_dataset(config["dataset_name"], split="train")
    first_img = np.array(dataset[0]["image"].convert("RGB"))
    first_seg = run_superpixel(first_img)

    plt.imsave("output/original.png", first_img)
    plt.imsave("output/superpixels.png", first_seg)

def preprocess_dataset_example():
    dataset = load_dataset(config["dataset_name"], split="train")
    dataset = dataset.train_test_split(test_size=10, train_size=10, seed=1)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=64,
        num_proc=config["num_proc"],
    )

    first_img = np.array(dataset["train"][0]["image"])
    first_seg = dataset["train"][0]["superpixels"]

    plt.imsave("output/original.png", first_img)
    plt.imsave("output/superpixels.png", first_seg)