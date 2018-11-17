# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from skimage.feature import corner_harris, corner_peaks
def getFeatures(image,bbox)
    F = image.shape[0]
    x1 = bbox[0,1,1]
    y1 = bbox[0,1,2]
    x2 = bbox[0,2,1]
    y2 = bbox[0,2,2]
    x3 = bbox[0,3,1]
    y3 = bbox[0,3,2]
    x1 = bbox[0,4,1]
    y1 = bbox[0,4,2]
    window = img[x1:x2,y2:y3,:]
    hight = window.shape[1]
    width = window.shape[2]
    window1 = corner_harris(window, method='k', k=0.05, eps=1e-06, sigma=1)
    coords = np.nonzero(window1>0.5)
    coords = np.nonzero(window1>0.5)
    #Threshold need to be tweaked
    x = coords[1]
    y = coords[0]
    N = x.shape[1]
    for i in range(F-2):
        x1 = bbox[i+1,1,1]
        y1 = bbox[i+1,1,2]
        x2 = bbox[i+1,2,1]
        y2 = bbox[i+1,2,2]
        x3 = bbox[i+1,3,1]
        y3 = bbox[i+1,3,2]
        x1 = bbox[i+1,4,1]
        y1 = bbox[i+1,4,2]
        window = img[x1:x2,y2:y3,:]
        hight = window.shape[0]
        width = window.shape[1]
        window1 = corner_harris(window, method='k', k=0.05, eps=1e-06, sigma=1)
        coords = np.nonzero(window1>0.5)
        #Threshold need to be tweaked
        x1 = coords[1]
        y1 = coords[0]
        n = x1.shape[1]
        maxn = maximum(n,N)
        x_temp = np.zeros((maxn,i+2))
        y_temp = np.zeros((maxn,i+2))
        x_temp[0:(N-1),0:i] = x
        y_temp[0:(N-1),0:i] = y
        x_temp[0:(n-1),i+1] = x1
        y_temp[0:(n-1),i+1] = y1
        x = x_temp
        y = y_temp
        N = x.shape[0]
        
    return x,y