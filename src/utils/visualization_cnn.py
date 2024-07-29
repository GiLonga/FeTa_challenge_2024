import os
from os.path import dirname, exists, join

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

#from src.config.constants import COLL_5_355, MODEL, RAW_DATA_DIR_PATH
#from src.data.processing import Processing
#from src.utils.local_optimization import Optimization
RAW_DATA_DIR_PATH=      r"/home/ubuntu/giorgio/v311/FeTa_challenge_2024/"


class Visualize:
    """Visualization class to visualize model's output"""

    def __init__(self) -> None:
        self.brains = np.load(join(RAW_DATA_DIR_PATH, "MRI_preprocessed_data.npy"))
        self.biometry = np.load(join(RAW_DATA_DIR_PATH, "biometry_data.npy")) #shape (N, 10, 4): 3 x,y,z coordinates and 1 label
        self.biometry=self.biometry[:,:,0:3]

    def reshape_output(
        self,
        y_hat: torch.Tensor,
        patient_idx: int,
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

    def plot_img(
        self,
        patient_idx: int,
        output: torch.Tensor,
        path: str,
        mse: torch.Tensor = torch.tensor(0),
    ) -> None:
        """
        Generates and saves a plot of images for a given patient.

        """

        
        if output.shape[0] ==30:
            output = self.reshape_output(output, patient_idx)


        # Retrieve information of the original shape

 #       local_optimization = Optimization(
 #           patient_idx=patient_idx,
 #           processing_output=processing_output,
 #           aspect_ratio=aspect_ratio,
 #       )
 #       local_optimization.optimize()

        self.single_figure_plot(
            patient_idx,
            path,
            output,
        )


    def single_figure_plot(
        self,
        patient_idx: int,
        path: str,
        reshaped_output: np.array,
    ) -> None:
        """
        Plot the predicted and true isocenters, jaws, and mask of a single patient, overlaying the predicted
        isocenters and jaws on the true data.

        Args:
        - patient_idx (int): Index of the patient to plot.
        - path (str): Path where to save the plot.
        - processing (Processing object): Processing object containing the true mask, isocenters, jaws, and
        collimator angles.
        - test (Processing object): Processing object containing the predicted isocenters, jaws, and collimator
        angles.
        - pix_spacing (float): Pixel spacing of the CT images.
        - slice_thickness (float): Slice thickness of the CT images.

        Returns:
        - None: The function saves the plot to disk, then closes it.
        """

        x_positions=reshaped_output[:,0]
        y_positions=reshaped_output[:,1]
        z_positions=reshaped_output[:,2]
        # Create a 1x3 grid of subplots

        x_positions_truth=self.biometry[patient_idx][:,0]
        y_positions_truth=self.biometry[patient_idx][:,1]
        z_positions_truth=self.biometry[patient_idx][:,2]

        color_dict = {
            0: 'red',
            1: 'green',
            2: 'cyan',
            3: 'magenta',
            4: 'orange'
        }

        bio_dict = {
            0: 'LCC',
            1: 'HV',
            2: 'bBIP_ax',
            3: ' sBIP_ax',
            4: 'TCD_cor'
        }


        # Display each image in a subplot
        for i in range(5):
            organ=bio_dict[i]
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(self.brains[patient_idx][0,int((x_positions[2*i]+x_positions[2*i+1])/2),:,:],cmap='gray')
            axs[0].set_title(f'Sagittal {organ}')
            axs[0].axis('off')  # Turn off axis
            axs[0].scatter(z_positions[2*i],y_positions[2*i], c=color_dict[i], s=4)
            axs[0].scatter(z_positions[2*i+1],y_positions[2*i+1], c=color_dict[i], s=4)
            axs[0].scatter(z_positions_truth[2*i],y_positions_truth[2*i], c="blue", s=4)
            axs[0].scatter(z_positions_truth[2*i+1],y_positions_truth[2*i+1], c="blue", s=4)

            axs[1].imshow(self.brains[patient_idx][0,:,int((y_positions[2*i]+y_positions[2*i+1])/2),:],cmap='gray')
            axs[1].set_title(f'Frontal {organ}')
            axs[1].axis('off')  # Turn off axis
            axs[1].scatter(z_positions[2*i],x_positions[2*i], c=color_dict[i], s=4)
            axs[1].scatter(z_positions[2*i+1],x_positions[2*i+1], c=color_dict[i], s=4)
            axs[1].scatter(z_positions_truth[2*i],x_positions_truth[2*i], c="blue", s=4)
            axs[1].scatter(z_positions_truth[2*i+1],x_positions_truth[2*i+1], c="blue", s=4)

            axs[2].imshow(self.brains[patient_idx][0,:,:,int((z_positions[2*i]+z_positions[2*i+1])/2)],cmap='gray')
            axs[2].set_title(f'Trasversal {organ}')
            axs[2].axis('off')  # Turn off axis
            axs[2].scatter(y_positions[2*i],x_positions[2*i], c=color_dict[i], s=4)
            axs[2].scatter(y_positions[2*i+1],x_positions[2*i+1], c=color_dict[i], s=4)
            axs[2].scatter(y_positions_truth[2*i],x_positions_truth[2*i], c="blue", s=4)
            axs[2].scatter(y_positions_truth[2*i+1],x_positions_truth[2*i+1], c="blue", s=4)


            # Display the plot

#        red_patch = mpatches.Patch(color="red", label="Pred")
#        blue_patch = mpatches.Patch(color="blue", label="Real")
#        plt.legend(handles=[red_patch, blue_patch], loc=0, frameon=True)

            eval_img_path = join(path, "img", "test")
            if not exists(eval_img_path):
                os.makedirs(eval_img_path)

            plt.savefig(join(eval_img_path, f"test_{patient_idx}_{i}"))
            plt.close()
