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
import csv
import fsl.data.image as fslimage
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform


sub_id='010' #To be changed
base_path_1=r"/home/ubuntu/giorgio/v311/FeTa_challenge_2024" #To be changed 
IMG256_PATH= os.path.join(base_path_1, f'sub-{sub_id}_rec-mial_T2w.nii.gz' ) #nmic
MODEL_PATH = r"/home/ubuntu/giorgio/v311/lightning_logs/brain_model/version_111/checkpoints/epoch=29-step=180.ckpt"
IMG_PATH = os.path.join(base_path_1, r"goundtruth_testBosisio/goundtruth", f'sub-{sub_id}_dseg_warped_0.5mm.nii.gz' )
MRI_PATH = os.path.join(base_path_1, r"MRI_testBosisio/images", f'sub-{sub_id}_brain_warped_0.5mm.nii.gz' )
TRANS_FILE = os.path.join(base_path_1, f'0GenericAffine.mat') #_{sub_id} 
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
    matrix = raw_img.affine
    raw_MRI = nib.load(MRI_PATH)
    my_MRI = raw_MRI.get_fdata()
    resized_img = resize(my_img)
    resized_MRI = resize(my_MRI)
    resized_img=np.expand_dims(resized_img, axis=0)
    resized_img=np.expand_dims(resized_img, axis=0)

    return resized_img, resized_MRI, matrix

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

def reconstruct_sparse_matrix(coordinates_and_values, num_x=176, num_y=224, num_z=176):
    # Initialize a 3D matrix with zeros
    sparse_matrix = np.zeros((num_x, num_y, num_z))
    
    # Populate the matrix using the coordinates_and_values array
    for coord in coordinates_and_values:
        x, y, z, value = coord
        sparse_matrix[int(x), int(y), int(z)] = value
    
    return sparse_matrix

def load_biometry(path) -> np.array:
    bio=[]
    my_img = nib.load(path)
    nii_data = my_img.get_fdata()
    non_zero_positions = np.nonzero(nii_data)
    values = nii_data[non_zero_positions]
    x_positions, y_positions, z_positions = non_zero_positions
    coordinates_and_values = np.column_stack((x_positions, y_positions, z_positions, values))
    coordinates_and_values = coordinates_and_values[coordinates_and_values[:, -1].argsort()]
    bio.append(coordinates_and_values)
    return np.array(bio[0])

def biometry_transformer(fixed_path:str, moving_path:str, matrix_path:str, output_path:str ) -> None:
    
    image = fslimage.Image(moving_path)
    data = image.data

    # Define the structuring element for dilatation and perform the dilatation
    dilated_data = ndimage.grey_dilation(data, footprint=np.ones((4,4,4)))

    dilated_image = fslimage.Image(dilated_data, header=image.header)
    dilated_image.save(output_path)

    fixed = ants.image_read(fixed_path)
    moving = ants.image_read( output_path)
    mywarpedimage = ants.apply_transforms(  fixed=fixed,
                                            moving=moving,
                                            transformlist= matrix_path,
                                            interpolator="nearestNeighbor",
                                            whichtoinvert=[True], 
                                            singleprecision=True,
                                            )
    
    ants.image_write(mywarpedimage, output_path)
    
def group_the_keypoints(array: np.array, threshold=3):
    def custom_distance(row1, row2, threshold=threshold):
        return 0 if np.all(np.abs(row1 - row2) <= threshold) else 1
    
    # Calculate the distance matrix
    distance_matrix = np.array([[custom_distance(row1, row2) for row2 in array] for row1 in array])
    # Perform hierarchical clustering
    Z = linkage(distance_matrix, method='complete')
    
    # Form clusters based on a threshold distance
    labels = fcluster(Z, t=2, criterion='maxclust')
    
    # Separate the rows into clusters
    cluster1 = array[labels == 1]
    cluster2 = array[labels == 2]
    
    # Calculate the averages of each cluster
    average_cluster1 = np.mean(cluster1, axis=0) if len(cluster1) > 0 else np.zeros(array.shape[1])
    average_cluster2 = np.mean(cluster2, axis=0) if len(cluster2) > 0 else np.zeros(array.shape[1])
    
    # Combine the averages into a single array
    label_1 = np.array([average_cluster1, average_cluster2])
    
    return label_1.astype(int)

if __name__ == "__main__":

    resized_img, resized_MRI, matrix = import_img(IMG_PATH, MRI_PATH) 
    lightning_cnn = LitCNN.load_from_checkpoint(checkpoint_path=MODEL_PATH)
    biometry=lightning_cnn(torch.Tensor(resized_img).cuda())
    biometry=reshape_output(biometry[0])
    opti= Optimization(resized_img, resized_MRI, biometry,)
    opti.optimize()
    biometry=opti.biometry
    original_img, biometry = inverse_resize(resized_img[0][0], biometry)

    # Here I'm creating the sparse matrix, alined as the patient matrix in the Atlas space.
    values=np.array([1,1,2,2,3,3,4,4,5,5])
    bio_val=np.column_stack((biometry, values))
    bio_sparse=reconstruct_sparse_matrix(bio_val)
    output=r"/home/ubuntu/giorgio/v311/FeTa_challenge_2024/inference_bio.nii.gz"
    img = nib.Nifti1Image(bio_sparse, affine=matrix)
    nib.save(img, output)

    f=  IMG256_PATH
    m=  os.path.join(base_path_1, f'sub-{sub_id}_bio.nii.gz' ) #To be changed
    mat = TRANS_FILE 
    biometry_transformer(f,m,mat,output)


    biometry = load_biometry(output)
    trasformed_biometry= np.empty((0, 4), dtype=int)
    for i in range(5):
        selected_rows = group_the_keypoints(biometry[biometry[:, 3] == i+1])
        trasformed_biometry=np.concatenate((trasformed_biometry, selected_rows), axis=0)
    biometry = trasformed_biometry[:,0:3]

    sparse256 = reconstruct_sparse_matrix(trasformed_biometry, num_x=256, num_y=256, num_z=256)

    raw_img = nib.load(IMG256_PATH)
    my_img = raw_img.get_fdata()
    matrix = raw_img.affine
    img = nib.Nifti1Image(sparse256, affine=matrix)
    nib.save(img, output)

    biometry_df = pd.DataFrame(biometry, columns=['x','y','z'])
    
    # Save coordinates of mapped keypoints using .csv file
    biometry_df.to_csv(os.path.join(OUT_PATH, "keypointsFinal_coords.csv"))    
    # For each couple of points measure their distance and put in csv file with appropriate label 

    print(biometry)
    bio_img = sitk.ReadImage(output)

    results = {'LCC': None, 'HV': None, 'bBIP': None, 'sBIP': None, 'TCD': None}
    res = pd.DataFrame()
    data_frames=[]

    for key in REGION_DICT:
        dist = get_dist(bio_img, key)
        print("The label ", key , "measures: ", dist)
        results[key] = dist

    data_frames.append(pd.DataFrame([results]))    
    res= pd.concat(data_frames, ignore_index=True)

    # Genereting CSV file name
    csv_file = os.path.join(OUT_PATH,'biometry.csv')
    res.to_csv(csv_file, index=False)

    print(f"Data written to {csv_file}")
