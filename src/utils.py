"""Where results are saved, and computing things only once."""
import os
import random

import numpy as np
import torch

DRIVE_ROOT = "/content/drive/MyDrive/sarcasm"


def mount_drive():
    """Mount Google Drive in Colab. Returns True if results will be saved on Drive."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        return True
    except Exception as e:  # not in Colab, or the user declined
        print("Google Drive not mounted, results stay local:", type(e).__name__)
        return False


def results_dir(sub=""):
    """Folder for results: on Google Drive if mounted, otherwise ./results/local."""
    base = DRIVE_ROOT if os.path.isdir("/content/drive/MyDrive") else os.path.join("results", "local")
    path = os.path.join(base, sub)
    os.makedirs(path, exist_ok=True)
    return path


def cached(name, compute, sub="", recompute=False):
    """Return the saved result `name` if it exists, else compute it, save it and return it.

    If Colab stops in the middle, rerunning the notebook skips everything already saved.
    """
    path = os.path.join(results_dir(sub), name + ".pt")
    if os.path.exists(path) and not recompute:
        print(f"[cached] {path}")
        return torch.load(path, weights_only=False)
    value = compute()
    torch.save(value, path)
    print(f"[saved]  {path}")
    return value


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
