import numpy as np
from skimage.feature import peak_local_max


def anms(cimg, max_pts):
    
    # Find local maxima (3*3 window)
    local_max = peak_local_max(cimg, min_distance=1, indices=False, threshold_abs=10e-3)
    coordinates = np.where(local_max == 1)
    Y = coordinates[0]
    X = coordinates[1]
    peaks = cimg[local_max]
    
    # Sort local maxima 
    order = np.argsort(peaks)[::-1]
    peaks = peaks[order]
    X = X[order]
    Y = Y[order]
    
    # Calculate radius for each local maxima
    r = np.zeros([len(peaks)], dtype=np.float)
    for i in range(len(peaks)):
        if i == 0:
            r[i] = max(cimg.shape)
            continue;
            
        distances = np.sqrt( (X[i]-X[:i])**2 + (Y[i]-Y[:i])**2 )
        r[i] = np.min(distances)
    
    # Sort by r
    order = np.argsort(r)[::-1]
    X = X[order][0:max_pts]
    Y = Y[order][0:max_pts]
    
    # Reshape x, y
    x = X.reshape([-1,1])
    y = Y.reshape([-1,1])
    
    return x, y