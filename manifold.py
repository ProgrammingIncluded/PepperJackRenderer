############################################
# Project: PepperJackRenderer
# File: manifold.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
# Description:
#     Stores the data to create the render manifold.
# Edit this file for desired render effects.
############################################

# Externals
import numpy as np
from scipy.spatial.transform import Rotation as R

# Geometry attributes
Z_ROTATION = 0
Y_ROTATION = -5
TRANSLATE = [0, 0, 25] # X, Y, Z where z represents the time/frame axis

## RENDER GEOMETRY PARAMETERS ##
def get_transform_matrix():
    """
        Transformation matrix used for rendering the viewport.
        Should transform from color space to camera space. In otherwords, imagine moving
        the camera using this transform matrix.
    """
    # Move the camera forward by this amount
    translate = translation_matrix(TRANSLATE)
    rotation_z = create_rotation("z", Z_ROTATION)
    rotation_z = to_homogenous(rotation_z)
    rotation_y = create_rotation("y", Y_ROTATION)
    rotation_y = to_homogenous(rotation_y)

    transformation_matrix = (translate @ rotation_z @ rotation_y)

    return transformation_matrix

def create_manifold(resolution_x, resolution_y):
    """Creates a 2D vertical grid of discrete points from the resolution."""
    result_x = []
    result_y = []
    result_z = []
    result_t = []
    for y in range(0, resolution_y):
        for x in range(0, resolution_x):
            result_x.append(x)
            result_y.append(y)
            result_z.append(0)
            result_t.append(1)
    return np.array([result_x, result_y, result_z, result_t])

## HELPER FUNCTIONS ##
def translation_matrix(v):
    return np.array([
        [1, 0, 0, v[0]],
        [0, 1, 0, v[1]],
        [0, 0, 1, v[2]],
        [0, 0, 0, 1]
    ])

def to_homogenous(m):
    result = np.concatenate((m, np.array([[0, 0, 0]])), axis=0)
    return np.concatenate((result, np.array([[0], [0], [0], [1]])), axis=1)

def create_rotation(axis, deg):
    return R.from_euler(axis, deg, degrees=True).as_dcm()

