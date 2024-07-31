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
import sys


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
        patient_mask: np.array,
        patient_MRI: np.array,
        biometry_output: np.array,
    ) -> None:
        self.brain_mask = patient_mask[0][0]
        self.brain_MRI = patient_MRI
        self.biometry = biometry_output

    def find_the_line(self,axis="trs"):
        if axis=="trs":
            points = [(self.biometry[self.organ+1,0],self.biometry[self.organ+1,1]),(self.biometry[self.organ,0],self.biometry[self.organ,1])] 
            x_coords, y_coords = zip(*points) #li zippo
            A = vstack([x_coords,ones(len(x_coords))]).T
            self.slope, self.intercept = lstsq(A, y_coords)[0]
            #print("Line Solution is y = {m}x + {c}".format(m = self.slope,c = self.intercept))
        elif axis=="sag":
            points = [(self.biometry[self.organ,2],self.biometry[self.organ,1]),(self.biometry[self.organ+1,2],self.biometry[self.organ+1,1])]
            x_coords, y_coords = zip(*points) #li zippo
            A = vstack([x_coords,ones(len(x_coords))]).T
            self.slope, self.intercept = lstsq(A, y_coords)[0]
            #print("Line Solution is y = {m}x + {c}".format(m = self.slope,c = self.intercept))

    def segment_LCC(self):
        organ=0
        np.set_printoptions(threshold=sys.maxsize)
        sag_img=self.brain_MRI[int((self.biometry[organ+1,2]+self.biometry[organ,2])/2),:,:]
        min_val = np.min(sag_img)
        max_val = np.max(sag_img)
        # Apply the min-max normalization formula
        normalized_array = (sag_img - min_val) / (max_val - min_val)
        #print(normalized_array)
        boundary=0.6
        self.LCC_mask = np.where(normalized_array < boundary, 0, 1)
        while (np.sum(self.LCC_mask)<2500) :
            boundary-=0.01
            self.LCC_mask = np.where(normalized_array < boundary, 0, 1)
        #while (np.sum(self.LCC_mask)>4200) :
        #    boundary+=0.01
        #    self.LCC_mask = np.where(normalized_array < boundary, 0, 1)
        #print(np.sum(self.LCC_mask))
        #plt.imshow(self.LCC_mask, cmap='gray')
        #plt.scatter(self.biometry[0,2], self.biometry[0,1], c="red", s=3)
        #plt.scatter(self.biometry[1,2], self.biometry[1,1],  c="red", s=3)
    
    def optimize_LCC(self):
        #while (self.LCC_mask[int(self.biometry[1,2]), int(self.biometry[1,1])]) == 0 :
        #        self.biometry[1,1]=+1
        #plt.imshow(self.LCC_mask, cmap='gray')
        #plt.scatter(self.biometry[0,2], self.biometry[0,1], c="red", s=3)
        #plt.scatter(self.biometry[1,2], self.biometry[1,1],  c="red", s=3)
        print("TO DO")

    def abline(self, axis="trs"):
        """Plot a line from slope and intercept"""
        if axis=="trs":
            #plt.imshow(self.brain_mask[:,:,int((self.biometry[self.organ+1,2]+self.biometry[self.organ,2])/2)], cmap='gray') #To check the index of biometry
            self.x_vals = range(128)
            self.y_vals = self.intercept + self.slope * self.x_vals
            #plt.plot( self.y_vals,self.x_vals, '--')
        elif axis=="sag":
            #plt.imshow(self.brain_mask[int((self.biometry[self.organ+1,0]+self.biometry[self.organ,0])/2),:,:], cmap='gray') #To check the index of biometry
            self.x_vals = range(128)
            self.y_vals = self.intercept + self.slope * self.x_vals
            #plt.plot( self.x_vals, self.y_vals, '--')

    def filter_points_by_mask(self, mask, label:int, axis="trs"):
        """Filter points that fall within the mask."""
        mask_points = []
        if axis=="sag":
            dummy=self.x_vals
            self.x_vals=self.y_vals
            self.y_vals=dummy
        for point_x, point_y in zip(self.x_vals, self.y_vals) :
            x, y = int(point_x), int(point_y)
            if mask[x, y]==label:  # Note: Mask indices should be in (row, col) format
                if axis=="sag":
                    mask_points.append((point_y,point_x))
                else:
                    mask_points.append((point_x,point_y))
        self.mask_points=np.array(mask_points)

    # TO DO try to figure out if points is an array or a list.
    def find_max_distance_points(self, orig_1, orig_2):
        """
        
        Find the two points that maximize the distance between them.

        """
        max_dist = 0
        max_points = (orig_1, orig_2)
        for i in range(len(self.mask_points)):
            for j in range(i + 1, len(self.mask_points)):
                dist = euclidean(self.mask_points[i], self.mask_points[j])
                if dist > max_dist:
                    max_dist = dist
                    max_points = (self.mask_points[i], self.mask_points[j])
        self.max_points=max_points


    def optimize(self):
        for i in range(5):
            if i == 1 :
                self.organ = 2
                self.find_the_line(axis="sag")
                self.abline(axis="sag")
                self.filter_points_by_mask(self.brain_mask[int((self.biometry[self.organ+1,0]+self.biometry[self.organ,0])/2),:,:], 5, axis="sag")
                Spoint_1=[self.biometry[self.organ,2] , self.biometry[self.organ,1]]
                Spoint_2=[self.biometry[self.organ+1,2] , self.biometry[self.organ+1,1]]
                self.find_max_distance_points(Spoint_1,Spoint_2)
                self.biometry[self.organ+1,2]=self.max_points[0][0]
                self.biometry[self.organ+1,1]=self.max_points[0][1]
                self.biometry[self.organ,2]=self.max_points[1][0]
                self.biometry[self.organ,1]=self.max_points[1][1]

            elif  i == 2 :
                self.organ=4
                self.find_the_line()
                self.abline()
                self.filter_points_by_mask(self.brain_mask[:,:,int((self.biometry[self.organ+1,2]+self.biometry[self.organ,2])/2)], 2)
                Spoint_1=[self.biometry[self.organ+1,0] , self.biometry[self.organ+1,1]]
                Spoint_2=[self.biometry[self.organ,0] , self.biometry[self.organ,1]]
                self.find_max_distance_points(Spoint_1,Spoint_2)
                self.biometry[self.organ+1,0]=self.max_points[0][0]
                self.biometry[self.organ+1,1]=self.max_points[0][1]
                self.biometry[self.organ,0]=self.max_points[1][0]
                self.biometry[self.organ,1]=self.max_points[1][1]
            elif i == 3 :
                self.organ=6
                self.find_the_line()
                self.abline()
                self.filter_points_by_mask(self.brain_mask[:,:,int((self.biometry[7,2]+self.biometry[6,2])/2)], 1)
                Spoint_1=[self.biometry[self.organ+1,0] , self.biometry[self.organ+1,1]]
                Spoint_2=[self.biometry[self.organ,0] , self.biometry[self.organ,1]]
                self.find_max_distance_points(Spoint_1,Spoint_2)
                self.biometry[self.organ+1,0]=self.max_points[0][0]
                self.biometry[self.organ+1,1]=self.max_points[0][1]
                self.biometry[self.organ,0]=self.max_points[1][0]
                self.biometry[self.organ,1]=self.max_points[1][1]
            elif i == 4 :
                #                                   TO DO: 
                # #This is optimized from a trasversal POV. It should be done from the Coronal
                self.organ=8
                self.find_the_line()
                self.abline()
                self.filter_points_by_mask(self.brain_mask[:,:,int((self.biometry[self.organ+1,2]+self.biometry[self.organ,2])/2)], 5)
                Spoint_1=[self.biometry[self.organ+1,0] , self.biometry[self.organ+1,1]]
                Spoint_2=[self.biometry[self.organ,0] , self.biometry[self.organ,1]]
                self.find_max_distance_points(Spoint_1,Spoint_2)
                self.biometry[self.organ+1,0]=self.max_points[0][0]
                self.biometry[self.organ+1,1]=self.max_points[0][1]
                self.biometry[self.organ,0]=self.max_points[1][0]
                self.biometry[self.organ,1]=self.max_points[1][1]
