import numpy as np
import cv2

# helper function
from anms import anms
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation


def objectTracking(rawVideo)

    # Initial tracking box; differ for each video
    bbox = np.zeros([1,4,2], dtype=np.int)
    # Initial box for Easy Video
    bbox[0,:,0] = [290, 400, 400, 290]
    bbox[0,:,1] = [180, 180, 265, 265]
    # Initial box for Medium Video
    # bbox[0,:,0] = [262, 355, 355, 262]
    # bbox[0,:,1] = [455, 455, 515, 515]

    # Read in original video
    cap = rawVideo
    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_num    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Parameter for our new video
    fps = 25
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('test_result.avi', fourcc, fps, (frame_width,frame_height),isColor=1)

    # Initialization with the first frame
    ret, frame = cap.read()
    I1 = frame
    g1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    startXs, startYs = getFeatures(g1 ,bbox)

    # For each frame after
    count = 0
    for i in range(frame_num-1):
        ret, frame = cap.read()
        if ret == True:
            # track features
            I2 = frame
            g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            newXs, newYs = estimateAllTranslation(startXs, startYs, I1, I2)
            startXs, startYs, newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)
            
            # draw new bounding box 
            newFrame = frame.copy()
            for j in range(newbbox.shape[0]):
                for k in range(startXs.shape[0]):
                    newFrame = cv2.circle(newFrame, (startXs[k,j], startYs[k,j]), 5, [255,255,0], -1)
                box = newbbox[j,:,:]
                box[:,0] = np.clip(box[:,0], 0, frame_width)  # make sure box corner is within image
                box[:,1] = np.clip(box[:,1], 0, frame_height)
                newFrame = cv2.polylines(newFrame, [box], True, [255,255,0], 3)
            
            # write the new frame
            out.write(newFrame)
            
            # stage for next frame
            I1 = I2
            bbox = newbbox
            count += 1

            # re-estimate features 
            if count == 8:
                count = 0
                startXs, startYs = getFeatures(g2 ,bbox)
            
        else:
            break

if __name__ = "__main__":
    rawVideo = cv2.VideoCapture('Easy.mp4')
    objectTracking(rawVideo)
    
    
    return out
