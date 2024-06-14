"""Dataset utility functions"""

from os.path import dirname, join

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

#from src.config.constants import INTERIM_DATA_DIR_PATH
#from src.data.augmentation import Augmentation

RAW_DATA_DIR_PATH=r"D:\Feta\FeTa_challenge_2024"

class Dataset:
    """Dataset class to load and stratify data"""

    def __init__(self) -> None:
        with np.load(join(RAW_DATA_DIR_PATH, "nii_data.npz")) as npz_brains:
            self.brains = list(npz_brains.values())
        self.normalize_brain_hu()
        self.num_patients = len(self.brains)

        #self.isocenters_pix = np.load(
        #    join(INTERIM_DATA_DIR_PATH, "isocenters_pix.npy")
        #)  # shape=(N, 12, 3)


    def normalize_brain_hu(self, background=0) -> None:
        """Normalize the mri corresponding to the density.

        Can be defined in a better way with the masks.
        """
        for i, brain in enumerate(self.brains):
            non_zero_values = brain[np.nonzero(brain)]
            min_value = np.min(non_zero_values) if background == 0 else np.min(brain)
            max_value = np.max(non_zero_values) if background == 0 else np.max(brain)
            difference = max_value - min_value
            normalized = (
                np.where(brain != 0, (brain - min_value) / difference, background)
                if background == 0
                else (brain - min_value) / difference
            )

            self.brains[i] = normalized


if __name__=="__main__":
    cervelli=Dataset()
    print("hello")