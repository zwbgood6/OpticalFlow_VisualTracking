import numpy as np
from scipy import interpolate


def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2)
    # (INPUT) startX: Represents the starting X coordinate
    # (INPUT) startY: Represents the starting Y coordinate
    
    # (OUTPUT) newX: Represents the new X coordinate
    # (OUTPUT) newY: Represents the new Y coordinate
    
    h, w = img1.shape()  
    
    # initial displacement and maximum iterations
    u = 0
    v = 0
    k = 10
    
    # initial feature point location in img2
    newX = startX + u
    newY = startY + v
    
    # interpolate version of img1, img2, Ix, Iy
    img1p = interpolate.interp2d(np.arange(w), np.arange(h), img1, kind='linear')
    img2p = interpolate.interp2d(np.arange(w), np.arange(h), img2, kind='linear')
    Ixp   = interpolate.interp2d(np.arange(w), np.arange(h), Ix, kind='linear')
    Iyp   = interpolate.interp2d(np.arange(w), np.arange(h), Iy, kind='linear')
    
    # optical flow
    for i in range(k):
        
        '''
        Still working on it
        '''
        
        
    return newX, newY