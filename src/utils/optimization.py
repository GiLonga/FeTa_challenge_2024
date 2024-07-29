from dataclasses import dataclass, field
from os.path import dirname, join

import numpy as np
import pandas as pd
#from gradient_free_optimizers import GridSearchOptimizer, ParallelTemperingOptimizer
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
#from skimage import io
from scipy.spatial.distance import euclidean
from numpy import ones,vstack
from numpy.linalg import lstsq


@dataclass
class OptimizationResult:
    """Result of the local optimization:
    'x' pixels of the maximum extension of the ribs and iliac crests.
    """

    y_voxel: int = field(default=0)
    x_voxel: int = field(default=0)
    z_voxel: int = field(default=0)




class Optimization:
    """Local optimization of the abdomen isocenter and related fields.
    The algorithm performs roughly the following steps:
    1) Approximate the location of the spine between the iliac crests and ribs.
    2) Within an appropriate neighborhood of this location, search the pixels
    corresponding to the maximum extension of the iliac crests and ribs.
    3) Adjust the abdomen isocenter and related fields according to the iliac crests
    and ribs positions.
    """

    def __init__(
        self,
        patient_mask: int,
        biometry_output: np.array,
    ) -> None:
        self.brain_mask=patient_mask
        self.biometry=biometry_output

    def find_the_line(self):
        points = [(self.biometry[5,0],self.biometry[5,1]),(self.biometry[4,0],self.biometry[4,1])] #the necessary biom.
        x_coords, y_coords = zip(*points) #li zippo
        A = vstack([x_coords,ones(len(x_coords))]).T
        self.slope, self.intercept = lstsq(A, y_coords)[0]
        print("Line Solution is y = {m}x + {c}".format(m = self.slope,c = self.intercept))

    def abline(self):
        """Plot a line from slope and intercept"""
        plt.imshow(self.brain_mask[:,:,self.biometry[3]], cmap='gray') #To check the index of biometry
        plt.xlim(0, 128)
        plt.ylim(0, 128)
        self.x_vals = range(128)
        self.y_vals = self.intercept + self.slope * self.x_vals
        plt.plot( self.y_vals,self.x_vals, '--')

    def filter_points_by_mask(self, mask, label:int):
        """Filter points that fall within the mask."""
        mask_points = []
        for point_x, point_y in zip(self.x_vals, self.y_vals) :
            x, y = int(point_x), int(point_y)
            if mask[x, y]==label:  # Note: Mask indices should be in (row, col) format
                mask_points.append((point_x,point_y))
        self.mask_points=np.array(mask_points)


    # TO DO try to figure out if points is an array or a list.
    def find_max_distance_points(self, orig_1, orig_2, points):
        """Find the two points that maximize the distance between them."""
        max_dist = 0
        max_points = (orig_1, orig_2)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = euclidean(points[i], points[j])
                if dist > max_dist:
                    max_dist = dist
                    max_points = (points[i], points[j])
        self.max_points=max_points



    def optimize(self):
        for i in range(4):
            if  i == 2 :
                self.find_the_line()
                self.abline()
                # TO DO 69 has to be removed.
                self.filter_points_by_mask(self.brain_mask[:,:,69], 2)
                # TO DO 
                self.find_max_distance_points([22,56],[105,55])
