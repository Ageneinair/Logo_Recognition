import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD,RMSprop
from keras.models import model_from_json
from model import def_model
from datetime import datetime

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

if __name__ == "__main__":
    data, labels = get_data()
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
  
    # save model
    time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model.save(time + "model.h5")
    print("Saved model to disk")

    # visualize training history
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(time + 'model_acc.png')
