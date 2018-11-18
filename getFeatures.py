import numpy as np
from skimage.feature import corner_harris

# helper function for ANMS
from anms import anms



def getFeatures(img,bbox):
    # (INPUT) img: H x W matrix representing grayscale image
    # (INPUT) bbox: F x 4 x 2 matrix representing coordinates of F objects 
    
    # (OUTPUT) x: N × F matrix representing the N row features coordinates of F objects
    # (OUTPUT) y: N × F matrix representing the N col features coordinates of F objects
    
    
    # number of object, and maximum number of feature points to track per object
    F = bbox.shape[0]
    N = 15
    
    # initialize x,y to store feature coordinates
    x = np.zeros([N, F], dtype=np.int)
    y = np.zeros([N, F], dtype=np.int)
    
    for i in range(F):
        # corner points
        x1, y1 = bbox[i,0,:]
        x2, y2 = bbox[i,1,:]
        x3, y3 = bbox[i,2,:]
        x4, y4 = bbox[i,3,:]
        
        # window of interest
        minX = np.min([x1,x2,x3,x4])
        maxX = np.max([x1,x2,x3,x4])
        minY = np.min([y1,y2,y3,y4])
        maxY = np.max([y1,y2,y3,y4])
        window = img[minY:maxY, minX:maxX]
        
        # detect feaure within this window
        cimg = corner_harris(window, k=0.05, eps=1e-06, sigma=1)
        xi, yi = anms(cimg, N)
        
        # threshold on the N feature points
        threhold = 0.005
        features = cimg[yi, xi]
        xi[features < threhold] = -1
        yi[features < threhold] = -1
        
        # re-arange xi, yi to put -1 to the end
        order = np.argsort(xi, axis=0)
        order = order[::-1].squeeze()
        xi = xi[order]
        yi = yi[order]
        
        # fill in final feature coodinates
        x[:,i] = xi
        y[:,i] = yi
    
    return x,y