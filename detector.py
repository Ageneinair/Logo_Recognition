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
WIN_STEP = 10
ROI_SIZE = (64, 64)

labels = {}

if __name__ == "__main__":
    # load model
    model = load_model('2019-07-22_02_54_28model.h5')

    img_file = "1.jpeg"
    orig = cv2.imread(img_file)
    # resize the input image to be a square
    resized = cv2.resize(orig, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

    # initialize the batch ROIs and (x, y)-coordinates
    batchROIs = None
    batchLocs = []
    # loop over the image pyramid
    for image in image_pyramid(resized, scale=PYR_SCALE,minSize=ROI_SIZE):
        # loop over the sliding window locations
        for (x, y, roi) in sliding_window(resized, WIN_STEP, ROI_SIZE):
            roi = roi/255
            roi = np.expand_dims(roi, axis=0)

            # if the batch is None, initialize it
            if batchROIs is None:
                batchROIs = roi

            # otherwise, add the ROI to the bottom of the batch
            else:
                batchROIs = np.vstack([batchROIs, roi])

            # add the (x, y)-coordinates of the sliding window to the batch
            batchLocs.append((x, y))


        # classify the batch, then reset the batch ROIs and
        # (x, y)-coordinates
        model.predict(batchROIs)
        labels = logo_prediction(model, batchROIs, batchLocs,labels, minProb=0.999999)

    # loop over the labels for each of detected objects in the image
    for k in labels.keys():
        # clone the input image so we can draw on it
        clone = resized.copy()

        # grab the bounding boxes and associated probabilities for each
        # detection, then apply non-maxima suppression to suppress
        # weaker, overlapping detections
        boxes = np.array([p[0] for p in labels[k]])
        proba = np.array([p[1] for p in labels[k]])
        boxes = non_max_suppression(boxes, proba, 0.8)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # show the output image
        print("[INFO] {}: {}".format(k, len(boxes)))
        cv2.imwrite('result.png',clone)
        #cv2.waitKey(0)

