import os
import cv2
import numpy as np
import imutils
from keras.models import load_model
from imutils.object_detection import non_max_suppression
from utils import sliding_window
from utils import image_pyramid
from utils import logo_prediction

INPUT_SIZE = (300, 300)
PYR_SCALE = 1.5
WIN_STEP = 32
ROI_SIZE = (64, 64)

labels = {}

if __name__ == "__main__":
    # load model
    model = load_model('2019-07-22_04_20_49model.h5')

    img_file = "demo/2.png"
    orig = cv2.imread(img_file)
    resized = cv2.resize(orig, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

    # initialize the batch ROIs and (x, y)-coordinates
    batchROIs = None
    batchLocs = []
    # loop over the image pyramid
    for image in image_pyramid(resized, scale=PYR_SCALE,minSize=ROI_SIZE):
        for (x, y, roi) in sliding_window(resized, WIN_STEP, ROI_SIZE):
            roi = roi/255
            roi = np.expand_dims(roi, axis=0)

            if batchROIs is None:
                batchROIs = roi

            else:
                batchROIs = np.vstack([batchROIs, roi])

            batchLocs.append((x, y))

        model.predict(batchROIs)
        labels = logo_prediction(model, batchROIs, batchLocs,labels, minProb=0.999999)

    clone = resized.copy()
    for k in labels.keys():
        if k == 'bkg':
            continue
        
        boxes = np.array([p[0] for p in labels[k]])
        proba = np.array([p[1] for p in labels[k]])
        print(boxes)
        print(proba)
        boxes = non_max_suppression(boxes, proba, overlapThresh=0)
        print(boxes)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(clone, k, (x1 + 6, y1 - 6), font, 0.5, (0, 0, 0), 2)
            
        # show the output image
        print("[INFO] {}: {}".format(k, len(boxes)))
        cv2.imwrite('result2.png',clone)
        #cv2.waitKey(0)

