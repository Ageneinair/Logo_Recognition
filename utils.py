import numpy as np
import imutils

CLASS_NAMES = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari',
            'Ford', 'Google', 'HP', 'Heineken', 'Intel', 'McDonalds', 'Mini', 'Nbc', 'Nike',
            'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef',
            'Vodafone', 'Yahoo']
 
def sliding_window(image, step, ws):
	# slide a window across the image
	for y in range(0, image.shape[0] - ws[1], step):
		for x in range(0, image.shape[1] - ws[0], step):
			# yield the current window
			yield (x, y, image[y:y + ws[1], x:x + ws[0]])

def image_pyramid(image, scale=1.5, minSize=(64, 64)):
	# yield the original image
	yield image

	# keep looping over the image pyramid
	while True:
		# compute the dimensions of the next image in the pyramid
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def logo_prediction(model, batchROIs, batchLocs, labels, minProb=0.5,dims=(64, 64)):
    preds = model.predict(batchROIs)
    for i in range(0,len(preds)):
        prob = np.max(preds[i])
        if prob > 0.5:
            index = np.argmax(preds[i])
            label = CLASS_NAMES[int(index)]
            # grab the coordinates of the sliding window for
            # the prediction and construct the bounding box
            (pX, pY) = batchLocs[i]
            box = (pX, pY, pX + dims[0], pY + dims[1])
            L = labels.get(label, [])
            L.append((box,prob))
            labels[label] = L
    return labels