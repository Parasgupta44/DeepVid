
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import math
import os
from glob import glob
from scipy import stats as s

import keras
from tensorflow.keras.models import Sequential,Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf

def ModelKi(location,BASE_DIR):
    print("first:" + location)
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(150,150,3))
	#base_model = InceptionV3(include_top=False, weights='imagenet',input_shape=(150,150,3))
    temploc = location.split('\\')
    print("------------------------|||||||||||")
    print(temploc)
    baseloc = ''
    for i in range(len(temploc)-1):

        baseloc += temploc[i] + '\\'
    print("third:" + baseloc)
    #print("fourth:"+location)
#defining the model architecture
    model = Sequential()
    model.add(Dense(1024, activation='relu',input_shape=(18432,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(101, activation='softmax'))

    model.load_weights(BASE_DIR + "\DV\models\weight_150_300_Incv3.hdf5")

    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# creating the tags
    train = pd.read_csv(BASE_DIR+'\DV\\train_new.csv')
    y = train['class']
    y = pd.get_dummies(y)

    pred = []
    cap = cv2.VideoCapture(location)
    print(cap)# capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    count = 0

    while(cap.isOpened()):

        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
        # storing the frames of this particular video in temp folder
            filename = str(baseloc) + 'temp\\' + "_frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()

    images = glob(str(baseloc) + "temp\*.jpg")

    prediction_images = []
    for i in range(len(images)):
        img = image.load_img(images[i], target_size=(150,150,3))
        img = image.img_to_array(img)
        img = img/255
        prediction_images.append(img)


# converting all the frames for a test video into numpy array
    prediction_images = np.array(prediction_images)
# extracting features using pre-trained model
    prediction_images = base_model.predict(prediction_images)
# converting features in one dimensional array
    prediction_images = prediction_images.reshape(prediction_images.shape[0], 3*3*2048)
# predicting tags for each array
    predictionar = model.predict_classes(prediction_images)

    predictionar = predictionar.reshape(predictionar.shape[0],1)

    mode = s.mode(predictionar)

    x = mode[0][0]

    ans = y.columns.values[x][0]

    files = glob(str(baseloc) + 'temp\*')
    for f in files:
        os.remove(f)
    return ans
