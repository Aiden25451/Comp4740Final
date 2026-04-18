config = {
    "dataset_name": "Simezu/brain-tumour-MRI-scan",
    "train_size": 4000,
    "test_size": 1500,
    "segments": 128,
    "batch_size": 32,
    "epochs": 60,
    "lr": 1e-3,

    # NEW
    "num_proc": 8,  # lower this if your PC is weak
    "superpixel_method": "felzenszwalb",  # "slic", "felzenszwalb", "quickshift"
}