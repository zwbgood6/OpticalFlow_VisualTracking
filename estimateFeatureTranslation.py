import numpy as np
from scipy import interpolate


def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    # (INPUT) startX: Represents the starting X coordinate
    # (INPUT) startY: Represents the starting Y coordinate
    
    # (OUTPUT) newX: Represents the new X coordinate
    # (OUTPUT) newY: Represents the new Y coordinate
    
    h, w = img1.shape
    
    # initial displacement and maximum iterations
    u = 0
    v = 0
    k = 20
    
    # initial feature point location in img2
    newX = startX + u
    newY = startY + v
    
    # interpolate version of img1, img2, Ix, Iy
    I1p = interpolate.interp2d(np.arange(w), np.arange(h), img1, kind='linear')
    I2p = interpolate.interp2d(np.arange(w), np.arange(h), img2, kind='linear')
    Ixp = interpolate.interp2d(np.arange(w), np.arange(h), Ix, kind='linear')
    Iyp = interpolate.interp2d(np.arange(w), np.arange(h), Iy, kind='linear')
    
    # points within 9*9 windows around startX and startY
    x1 = np.arange(startX-4, startX+5)
    y1 = np.arange(startY-4, startY+5)
    xx1, yy1 = np.meshgrid(x1, y1)
    ind_1 = np.stack([xx1,yy1], 2)
    ind_1 = ind_1.reshape((-1,2))
    
    # Ix and Iy values within img1 window
    Ix_window = [int(Ixp(ind[0], ind[1])) for ind in ind_1]
    Iy_window = [int(Iyp(ind[0], ind[1])) for ind in ind_1]
    sumIxIx = np.sum([i*i for i in Ix_window])
    sumIyIy = np.sum([i*i for i in Iy_window])
    sumIxIy = np.sum([i*j for i,j in zip(Ix_window, Iy_window)])
    A = np.array([[sumIxIx, sumIxIy], 
                  [sumIxIy, sumIyIy]])
    
    # optical flow
    for i in range(k):
        # points within 9*9 windows around newX and newY;
        # newX and newY change in each iteration so window need to be recalculated
        x2 = np.arange(newX-4, newX+5)
        y2 = np.arange(newY-4, newY+5)
        xx2, yy2 = np.meshgrid(x2, y2)
        ind_2 = np.stack([xx2,yy2], 2)
        ind_2 = ind_2.reshape((-1,2))
        
        # It = img2 - img1 within the 2 matching windows
        I1_window = [int(I1p(ind[0], ind[1])) for ind in ind_1]
        I2_window = [int(I2p(ind[0], ind[1])) for ind in ind_2]
        It = [i-j for i,j in zip(I2_window,I1_window)]
        sumIxIt = np.sum([i*j for i,j in zip(Ix_window, It)])
        sumIyIt = np.sum([i*j for i,j in zip(Iy_window, It)])
        b = -np.array([[sumIxIt],
                       [sumIyIt]])
        
        # d = A\b
        d = np.dot(np.linalg.inv(A), b)
        u = d[0]
        v = d[1]
        
        # update newX and newY
        newX = newX + u
        newY = newY + v

        # check if [u,v] have already converge
        if abs(u)<0.1 and abs(v)<0.1:
            break;
        
    
    return newX, newY
