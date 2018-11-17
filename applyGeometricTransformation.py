# -*- coding: utf-8 -*-
"""
Function: Transform the four corners of the bounding box from one frame to another.

@author: Wenbo Zhang
"""
import numpy as np
from skimage import transform as tf


def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
     
    # (INPUT) startXs: N × F matrix 
    # (INPUT) startYs: N × F matrix 
    # (INPUT) newXs: N × F matrix 
    # (INPUT) newYs: N × F matrix 
    # (INPUT) bbox: F × 4 × 2 matrix 
    
    # (OUTPUT) Xs: N1 × F matrix
    # (OUTPUT) Ys: N1 × F matrix
    # (OUTPUT) newbbox: F × 4 × 2 matrix
    
    # (PARAMETER) N: Number of features in an object
    # (PARAMETER) F: Number of objects you would like to track
    
    # Initialization
    N, F = startXs.shape
    count = 0
    Xs = np.zeros([N,F], dtype=np.int)
    Ys = np.zeros([N,F], dtype=np.int)
    startXsTemp = np.zeros([N,F], dtype=np.int)
    startYsTemp = np.zeros([N,F], dtype=np.int)
    newbbox = np.zeros([F, 4, 2], dtype=np.int)
    
    # Calculate matrix difference
    diffx = newXs - startXs
    diffy = newYs - startYs
    matrixDistance = np.sqrt(diffx^2 + diffy^2)
    correspondPointDistantThreshold = 4
    
    # Delete feature points whose distances are less than threshold 
    for i in range(F):
        for j in range(N):
            if matrixDistance[i][j] < correspondPointDistantThreshold:
                Xs[i][count] = newXs[i][j]
                Ys[i][count] = newYs[i][j]
                startXsTemp[i][count] = startXs[i][j]
                startYsTemp[i][count] = startYs[i][j]
                count += 1
            else:
                continue
    
    # Resize output variables
    Xs = Xs[:][:count]
    Ys = Ys[:][:count]
    startXsTemp = startXsTemp[:][:count]
    startYsTemp = startYsTemp[:][:count]
    
    # Calculate similarity matrix and new boundary box
    for k in range(F):
        src = np.concatenate((startXsTemp[k][:], startYsTemp[k][:]), axis=1)
        dst = np.concatenate((Xs[k][:], Ys[k][:]), axis=1)
        tform = tf.estimate_transform('similarity', src, dst)
        tformp = np.asmatrix(tform.params)
        newbbox[k][:][:] = tformp.dot(bbox[k][:][:])
    
    
    return Xs, Ys, newbbox