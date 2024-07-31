# Load
import torch
from lightning import Trainer
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import zoom
from src.data.dataset import Dataset
from src.modules.lightning_cnn import LitCNN
from src.utils.optimization import Optimization
import nibabel as nib
from torch.utils.data import TensorDataset
import pandas as pd
import ants
import os
import SimpleITK as sitk

MODEL_PATH = r"/home/ubuntu/giorgio/v311/lightning_logs/brain_model/version_111/checkpoints/epoch=29-step=180.ckpt"
IMG_PATH = r"/home/ubuntu/giorgio/v311/FeTa_challenge_2024/goundtruth_testBosisio/goundtruth/sub-010_dseg_warped_0.5mm.nii.gz"
MRI_PATH = r"/home/ubuntu/giorgio/v311/FeTa_challenge_2024/MRI_testBosisio/images/sub-010_brain_warped_0.5mm.nii.gz"
TRANS_FILE = r"/home/ubuntu/giorgio/v311/FeTa_challenge_2024/0GenericAffine.mat"
OUT_PATH = r"/home/ubuntu/giorgio/v311/FeTa_challenge_2024"

REGION_DICT = {
    "LCC": 1,
    "HV": 2,
    "bBIP": 3,
    "sBIP": 4,
    "TCD": 5,
    }

def resize(array: np.array,) :
    target_shape = (128, 128, 128)

    # Calculate zoom factors
    original_shape = array.shape
    zoom_factors = [t / o for t, o in zip(target_shape, original_shape)]

    # Resize the array using nearest-neighbor interpolation
    resized_array = zoom(array, zoom_factors, order=0)  # 'order=0' for nearest-neighbor interpolation
    return resized_array

def inverse_resize(resized_array: np.array, keypoints: np.array, original_shape=(176,224,176)):
    # Calculate the inverse zoom factors
    zoom_factors = [o / t for t, o in zip(resized_array.shape, original_shape)]

    # Resize the array back to the original shape using nearest-neighbor interpolation
    original_array = zoom(resized_array, zoom_factors, order=0)  # 'order=0' for nearest-neighbor interpolation

    transformed_keypoints = np.array([(int(x * zoom_factors[0]), int(y * zoom_factors[1]), int(z * zoom_factors[2])) for x, y, z in keypoints])
    
    return original_array, transformed_keypoints


def reshape_output(
    y_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Reshape the output tensor (1,30) of the model to the (10,3) model.

    Parameters:
    - y_hat (torch.Tensor): Predicted tensor (1,30) from the model.

    Returns:
    - torch.Tensor: output tensor with resized shape.

    """
    y_hat_new = y_hat.cpu().data.numpy()
    y_hat_new = y_hat_new.reshape(10,3)

    return y_hat_new

def import_img(IMG_PATH:str, MRI_PATH:str):
    
    raw_img = nib.load(IMG_PATH)
    my_img = raw_img.get_fdata()
    raw_MRI = nib.load(MRI_PATH)
    my_MRI = raw_MRI.get_fdata()
    resized_img = resize(my_img)
    resized_MRI = resize(my_MRI)
    resized_img=np.expand_dims(resized_img, axis=0)
    resized_img=np.expand_dims(resized_img, axis=0)

    return resized_img, resized_MRI

def get_dist(im, region):
    """
    Get the distance between the two points of a given region.
    """
    x, y, z = np.where(sitk.GetArrayFromImage(im) == REGION_DICT[region])
    if len(x) == 0:
        return np.nan
    p1 = np.array([x[0], y[0], z[0]])
    p2 = np.array([x[1], y[1], z[1]])
    ip_res = im.GetSpacing()[0]
    assert len(x) == 2, f"Region {region} has {len(x)} points"
    dist = round(np.linalg.norm((p1 - p2) * ip_res), 2)
    return dist

if __name__ == "__main__":

    resized_img, resized_MRI = import_img(IMG_PATH, MRI_PATH) 
    lightning_cnn = LitCNN.load_from_checkpoint(checkpoint_path=MODEL_PATH)
    biometry=lightning_cnn(torch.Tensor(resized_img).cuda())
    biometry=reshape_output(biometry[0])
    opti= Optimization(resized_img, resized_MRI, biometry,)
    opti.optimize()
    biometry=opti.biometry
    original_img, biometry = inverse_resize(resized_img[0][0], biometry)

    # Dataframe for mapping keypoints to image space
    biometry_df = pd.DataFrame(biometry, columns=['x','y','z'])
    # Apply inverse transform using "apply_transforms_to_points"
    biometryTrans_df = ants.apply_transforms_to_points(3, biometry_df, [TRANS_FILE], whichtoinvert=[True], verbose=False)
    # Save coordinates of mapped keypoints using .csv file
    biometryTrans_df.to_csv(os.path.join(OUT_PATH, "keypointsFinal_coords.csv"))    
    # For each couple of points measure their distance and put in csv file with appropriate label 
    
    
    # Save measures dataframe
   
    print( biometry)
