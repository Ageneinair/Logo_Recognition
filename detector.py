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
    model = load_model('2019-07-21_01:26:40model.h5')

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
            # take the ROI and pre-process it so we can later classify the
            # region with Keras
            #roi = img_to_array(roi)
            roi = roi/255
            roi = np.expand_dims(roi, axis=0)
            # roi = imagenet_utils.preprocess_input(roi)

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

        # loop over all bounding boxes for the label and draw them on
        # the image
    # 	for (box, prob) in labels[k]:
    # 		(xA, yA, xB, yB) = box
    # 		cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # 	# show the image *without* apply non-maxima suppression
    # 	cv2.imshow("Without NMS", clone)
    # 	clone = resized.copy()

        # grab the bounding boxes and associated probabilities for each
        # detection, then apply non-maxima suppression to suppress
        # weaker, overlapping detections
        boxes = np.array([p[0] for p in labels[k]])
        proba = np.array([p[1] for p in labels[k]])
        boxes = non_max_suppression(boxes, proba)

        # loop over the bounding boxes again, this time only drawing the
        # ones that were *not* suppressed
        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 0, 255), 2)

        # show the output image
        print("[INFO] {}: {}".format(k, len(boxes)))
        cv2.imwrite('result.png',clone)
        #cv2.waitKey(0)

