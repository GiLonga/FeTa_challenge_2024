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

MODEL_PATH = r"/home/ubuntu/giorgio/v311/lightning_logs/brain_model/version_111/checkpoints/epoch=29-step=180.ckpt"
IMG_PATH = r"/home/ubuntu/giorgio/v311/FeTa_challenge_2024/goundtruth_testBosisio/goundtruth/sub-010_dseg_warped_0.5mm.nii.gz"

def resize(array: np.array,) :
    target_shape = (128, 128, 128)

    # Calculate zoom factors
    original_shape = array.shape
    zoom_factors = [t / o for t, o in zip(target_shape, original_shape)]

    # Resize the array using nearest-neighbor interpolation
    resized_array = zoom(array, zoom_factors, order=0)  # 'order=0' for nearest-neighbor interpolation
    return resized_array

def reshape_output(
    y_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Reshape the output tensor of the 5_355 model to the 90 model.

    Parameters:
    - y_hat (torch.Tensor): Predicted tensor from the 5_355 model.
    - patient_idx (int): Index of the patient in the test set, to be reshaped.

    Returns:
    - torch.Tensor: output tensor with resized shape.

    """
    y_hat_new = y_hat.cpu().data.numpy()
    y_hat_new = y_hat_new.reshape(10,3)

    return y_hat_new

def import_img(IMG_PATH:str,):
    raw_img = nib.load(IMG_PATH)
    my_img = raw_img.get_fdata()
    resized_img = resize(my_img)
    resized_img=np.expand_dims(resized_img, axis=0)
    resized_img=np.expand_dims(resized_img, axis=0)
# Verify the result
    print(resized_img.shape)
    return resized_img



if __name__ == "__main__":

    resized_img = import_img(IMG_PATH) 

    (brains,
    y_reg,
    predict_index,
    ) = tuple(
        map(
            torch.Tensor,
            (
                resized_img,
                [np.ones(30)],
                1,
            ),
        )
    )
    pred_dataset = TensorDataset(
        brains,
        y_reg,
        predict_index,
    )

    dataset = pred_dataset
    lightning_cnn = LitCNN.load_from_checkpoint(checkpoint_path=MODEL_PATH)

    prediction_loader = DataLoader(
        dataset,
        num_workers=0,
    )

    biometry=trainer.predict(lightning_cnn, dataloaders=prediction_loader, return_predictions=True)
    biometry=reshape_output(biometry[0])



    trainer = Trainer(logger=False)

    print("The results are:",  biometry, "patients")