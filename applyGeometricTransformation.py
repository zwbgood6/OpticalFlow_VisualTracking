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
    matrixDistance = np.sqrt(diffx**2 + diffy**2)
    correspondPointDistantThreshold = 4
    
    # Delete feature points whose distances are less than threshold
    for j in range(F):
        for i in range(N):
            if matrixDistance[i][j] < correspondPointDistantThreshold:
                Xs[count][j] = newXs[i][j]
                Ys[count][j] = newYs[i][j]
                startXsTemp[count][j] = startXs[i][j]
                startYsTemp[count][j] = startYs[i][j]
                count += 1
        count = 0
    
    # Resize output variables
    maxCount = np.max(sum(matrixDistance < correspondPointDistantThreshold))
    Xs = Xs[:maxCount][:]
    Ys = Ys[:maxCount][:]
    startXsTemp = startXsTemp[:maxCount][:]
    startYsTemp = startYsTemp[:maxCount][:]
    
    # Trim and resize
    for k in range(F):
        X = np.trim_zeros(Xs[:,k], trim='b')
        Y = np.trim_zeros(Ys[:,k], trim='b')
        startX = startXsTemp[:len(X),k]
        startY = startYsTemp[:len(X),k]
        
        # bounding box
        src = np.vstack([startX, startY]).T
        dst = np.vstack([X, Y]).T
        x,y,w,h = cv2.boundingRect(dst)
        offset  = 8
        Xbox = [x-offset, x+w+2*offset, x+w+2*offset, x-offset]
        Ybox = [y-offset, y-offset,     y+h+2*offset, y+h+2*offset]
        
        newbbox[k,:,:] = np.vstack([Xbox, Ybox]).T
        #tform = tf.estimate_transform('similarity', src, dst)
        #box = tform(bbox[k,:,:])
        #newbbox[k,:,:] = box
    
    
    return Xs, Ys, newbbox