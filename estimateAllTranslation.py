import numpy as np
from scipy import signal


def estimateAllTranslation(startXs,startYs,img1,img2)
    # (INPUT) startXs: N × F matrix 
    # (INPUT) startYs: N × F matrix 
    
    # (OUTPUT) newXs: N × F matrix
    # (OUTPUT) newYs: N × F matrix
    
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])
    
    Ix = signal.convolve(img1, dx, mode='same')
    Iy = signal.convolve(img1, dy, mode='same')
    
    N, F = startXs.shape()
    newXs = np.zeros([N,F], dtype=np.int)
    newYs = np.zeros([N,F], dtype=np.int)
    
    for j in range(F):
        for i in range(N):
            startX = startXs[i,j]
            startY = startYs[i,j]
            newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2)
            newXs[i,j] = newX
            newYs[i,j] = newY
            
    return newXs, newYs