import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K
from keras.optimizers import Adam,SGD,RMSprop


HEIGHT, WIDTH = (64, 64)

def get_data():
    data = []
    labels = []
    for img in os.listdir("./data/new_dataset/"):
        img_file = cv2.imread(os.path.join("./data/new_dataset/",img))
        data.append(img_file)
        labels.append(img.split("_")[1].split(".")[0])
    data = np.stack(data)
    labels = np.stack(labels)
    # generate OneHot encoding
    le = LabelBinarizer()
    labels = le.fit_transform(labels)
    
    return data/255, labels

def def_model():
    model = Sequential()
    inputShape = (HEIGHT, WIDTH, 3)

    model.add(Conv2D(16, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Dense(len(CLASSNAME)))
    model.add(Activation("softmax"))

    return(model)

if __name__ == "__main__":
    data, lables = get_data()
    X,testX,y,testy = train_test_split(data, labels,test_size=0.1,stratify=labels,random_state=42)
    # data augmentation
    aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
        horizontal_flip=True, fill_mode="nearest")
    gen_flow=aug.flow(X, y,batch_size=64,seed=0)
    validation=aug.flow(testX,testy,batch_size=32,seed=0)
    
    model = def_model()
    # set optimizer
    opt = RMSprop(lr=0.001, rho=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

    history=model.fit_generator(gen_flow,
                                steps_per_epoch=len(X) // 32,
                                validation_data=validation,
                                validation_steps=len(testX) // 32,
                                epochs=100,
                                verbose=1)
    
