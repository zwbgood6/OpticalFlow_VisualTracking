import numpy as np
from scipy import signal


def estimateAllTranslation(startXs,startYs,img1,img2):
    # (INPUT) startXs: N × F matrix 
    # (INPUT) startYs: N × F matrix 
    # (OUTPUT) newXs: N × F matrix
    # (OUTPUT) newYs: N × F matrix
    
    # size of image
    H, W = img1.shape[0:2]
    I1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    I2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # derivative kernel
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])
    
    # gradient
    Ix = signal.convolve(I1, dx, mode='same')
    Iy = signal.convolve(I1, dy, mode='same')
    
    # newXs and newYs to store new locations
    N, F = startXs.shape
    newXs = np.zeros([N,F], dtype=np.int)
    newYs = np.zeros([N,F], dtype=np.int)
    
    # Iterate through each feature locations
    for j in range(F):
        for i in range(N):
            startX = startXs[i,j]
            startY = startYs[i,j]
            if startX == 0 or startY == 0:
                newXs[i,j] = 0
                newYs[i,j] = 0
            else:
                newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2)
                if newX > W or newY >H:  # transport out of image
                    newXs[i,j] = 0
                    newYs[i,j] = 0
                else:
                    newXs[i,j] = newX
                    newYs[i,j] = newY
    
    
    return newXs, newYs