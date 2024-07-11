"""Dataset utility functions"""

from os.path import dirname, join
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

#from src.config.constants import INTERIM_DATA_DIR_PATH
#from src.data.augmentation import Augmentation

RAW_DATA_DIR_PATH=      r"/home/ubuntu/giorgio/v311/FeTa_challenge_2024/"
INTERIM_DATA_DIR_PATH=  r"D:\Feta\Institution 2 - Medical University of Vienna\ "   #Actually this isn't working.

class Dataset:
    """Dataset class to load and stratify data"""

    def __init__(self) -> None:
        self.brains = np.load(join(RAW_DATA_DIR_PATH, "segment.npy"))
        self.normalize_brain_hu()
        self.num_patients = self.brains.shape[0]

        self.biometry = np.load(join(RAW_DATA_DIR_PATH, "biometry_data.npy")) #shape (N, 10, 4): 3 x,y,z coordinates and 1 label
        self.biometry=self.biometry[:,:,0:3]
    
        self.df_patient_info =pd.read_excel(join(dirname(RAW_DATA_DIR_PATH), r"datiFetal_Vienna.xlsx"))
        #self.indexes=self.df_patient_info.Subject
        self.indexes=np.arange(self.num_patients)

    def normalize_brain_hu(self, background=0) -> None:
        """Normalize the mri corresponding to the density.

        Can be defined in a better way with the masks.
        """
        for i, brain in enumerate(self.brains[0]):
            non_zero_values = brain[np.nonzero(brain)]
            min_value = np.min(non_zero_values) if background == 0 else np.min(brain)
            max_value = np.max(non_zero_values) if background == 0 else np.max(brain)
            difference = max_value - min_value
            normalized = (
                np.where(brain != 0, (brain - min_value) / difference, background)
                if background == 0
                else (brain - min_value) / difference
            )

            self.brains[0,i] = normalized


    def train_val_test_split(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into training, validation, and test sets.

        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray): Index splits for train, validation, and test sets.

        Notes:
            - The default split ratio is 80% for training, 10% for validation, and 10% for testing.
        """

        _, test_idx = train_test_split(
            self.indexes,
            train_size=0.91,
            random_state=42,
        )
        test_idx = np.array(test_idx)
        #train_idx = self.indexes[
        #    ~self.indexes.isin(
        #        test_idx
        #    )  # remove test_idx from data frame
        #].to_numpy()
        train_pos = np.ones(self.num_patients, dtype=bool)
        train_pos[test_idx] = False
        train_idx=self.indexes[train_pos]

        train_idx, val_idx = train_test_split(
            train_idx,
            train_size=0.9,
            random_state=42,
        )

        self.train_idx = train_idx
        return (train_idx, val_idx, test_idx)
    

    def get_dataset(self) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
        """
        Prepares and returns the three datasets needed for training a model from:
            - Input data (X): Array of pointclouds of shape (D, H, W).   THIS IS TO BE CHECKED
            - Target data (y_reg): Array of unique outputs derived from isocenters and jaws positions.


        Returns:
            tuple[TensorDataset, TensorDataset, TensorDataset]: A tuple containing:
                - Train dataset: TensorDataset for training of the model.
                - Validation dataset: TensorDataset for validation of the model.
                - Test dataset: TensorDataset for testing of the model.
        """
        biometry_pix_flat = self.biometry.reshape(self.num_patients,1,30)
        y_reg= self.unique_output(biometry_pix_flat)
        train_index, val_idx, test_index = self.train_val_test_split()

        (
            brain_train,
            y_reg_train,
            brain_val,
            y_reg_val,
            brain_test,
            y_reg_test,
            test_idx,
            train_index,
        ) = tuple(
            map(
                torch.Tensor,
                (
                    self.brains[train_index],
                    y_reg[train_index],
                    self.brains[val_idx],
                    y_reg[val_idx],
                    self.brains[test_index],
                    y_reg[test_index],
                    test_index,
                    train_index,
                ),
            )
        )

        train_dataset = TensorDataset(
            brain_train,
            y_reg_train,
        )

        val_dataset = TensorDataset(
            brain_val,
            y_reg_val,
        )

        test_len = test_index.shape[0]
        test_dataset = TensorDataset(
            brain_test,
            y_reg_test,
            test_idx,
            brain_train[0:test_len],  # mask_train = [0:11] or [0:3]
            train_index[0:test_len],  # mask_train = [0:11] or [0:3]
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
        )
 
    def prediction_dataset(self) -> TensorDataset:
        """
        Prepares and returns a dataset to evaluate the model's predictions.

        Returns:
            TensorDataset: The TensorDataset containing the data to evaluate the model.
        """

        biometry_pix_flat = self.biometry.reshape(self.num_patients,1,30)
        y_reg= self.unique_output(biometry_pix_flat)

        _, _, predict_index = self.train_val_test_split()

        (
            brains,
            y_reg,
            predict_index,
        ) = tuple(
            map(
                torch.Tensor,
                (
                    self.brains[predict_index],
                    y_reg[predict_index],
                    predict_index,
                ),
            )
        )

        pred_dataset = TensorDataset(
            brains,
            y_reg,
            predict_index,
        )

        return pred_dataset

    def unique_output(
        self, biometry_pix_flat) -> np.ndarray:
        """
        Create the target array with minimum elements (only unique information).

        Args:
            isocenters_pix_flat (np.ndarray): Flat array containing the isocenter values.

        Returns:
            np.ndarray: Array with the unique values from the input data.
                The resulting array has a shape of (self.num_patients, 1, 39).

        Notes:
            - The resulting array contains 8 values for the isocenters,
            21 values for the X_jaws, and 10 values for the Y_jaws.
            - Specific indices are used to select the unique values from the input arrays.
            Details about the selected indices can be found in the function implementation.
            6*5=30
        """
        #TO DO MAYBE
        y_reg=biometry_pix_flat
        return y_reg

if __name__=="__main__":
    cervelli=Dataset()
    print("The data are imported correctly, there are",  len(cervelli.brains), "patients")