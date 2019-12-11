<<<<<<< HEAD
#import the required libraries
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm



# open the .txt file which have names of training videos
f = open("trainlist01.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
train = pd.DataFrame()
train['video_name'] = videos
train = train[:-1]
train.head()



# open the .txt file which have names of test videos
f = open("testlist01.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
test = pd.DataFrame()
test['video_name'] = videos
test = test[:-1]
test.head()



# creating tags for training videos
train_video_tag = []
for i in range(train.shape[0]):
    train_video_tag.append(train['video_name'][i].split('/')[0])

train['tag'] = train_video_tag

# creating tags for test videos
test_video_tag = []
for i in range(test.shape[0]):
    test_video_tag.append(test['video_name'][i].split('/')[0])

test['tag'] = test_video_tag


# storing the frames from training videos
for i in tqdm(range(train.shape[0])):
    count = 0
    videoFile = train['video_name'][i]
    i = 0
    path = ""
    while(True):
        path += videoFile[i]
        i = i + 1
        if(videoFile[i] == '/'):
            break
    cap = cv2.VideoCapture('Videos/'+ path + '/'+ videoFile.split(' ')[0].split('/')[1])   # capturing the video from the given path
    #print(cap.grab())
    #path should be given accordingly
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder named train_1
            filename ='train_1/' + videoFile.split('/')[1].split(' ')[0] +"_frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()


# getting the names of all the images
images = glob("train_1/*.jpg")
train_image = []
train_class = []
for i in tqdm(range(len(images))):
    # creating the image name
    p1 = images[i].split('/')[0]
    p1 = p1[8:]
    train_image.append(p1)
    # creating the class of image
    i = 2
    p2 = ""
    while(True):
        if(p1[i] == '_'):
            break
        p2 = p2 + p1[i]
        i = i + 1
    train_class.append(p2)

# storing the images and their class in a dataframe
train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

# converting the dataframe into csv file
train_data.to_csv('train_new.csv',header=True, index=False)



#import libraries
import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras import backend as K




train = pd.read_csv('train_new.csv')
train.head()

# creating an empty list
train_image = []

# for loop to read and store frames
for i in tqdm(range(train.shape[0])):

    # loading the image and keeping the target size as (224,224,3)(size varies according to requirement)
    img = image.load_img('train_1/'+train['image'][i], target_size=(200,200,3))

    # converting it to array
    img = image.img_to_array(img)

    # normalizing the pixel value
    img = img/255

    # appending the image to the train_image list
    train_image.append(img)

# converting the list to numpy array
X = np.array(train_image)

# shape of the array
print(X.shape)

# separating the target
y = train['class']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)


# creating dummies of target variable for train and validation set
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

import tensorflow as tf

# creating the base model of pre-trained models
#base_model = VGG16(weights='imagenet', include_top=False,input_shape=(100,100,3))
base_model = tf.keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet',  input_shape=(200,200,3))
#base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',  input_shape=(200,200,3))
#base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3))


# extracting features for training frames
X_train = base_model.predict(X_train)
X_train.shape


# extracting features for validation frames
X_test = base_model.predict(X_test)
X_test.shape

print(X_train.shape[0])


# reshaping the training as well as validation frames in single dimension,acc to size of X_train after prediction
X_train = X_train.reshape(59075, 6*6*1024)
X_test = X_test.reshape(14769, 6*6*1024)

# normalizing the pixel values
max = X_train.max()
print(max)
X_train = X_train/max
X_test = X_test/max

print(X_train.shape)


#defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu',input_shape=(36864,)))#size filled according to X_train size after prediction
model.add(Dropout(0.5))                                       #Dropout to reduce overfitting
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(101, activation='softmax'))

# defining a function to save the weights of best model
from tensorflow.keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint('weight_200_200_dense.hdf5', save_best_only=True, monitor='val_loss', mode='min')


# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


y_train = y_train.to_numpy(copy=False)
y_test = y_test.to_numpy(copy=False)


#matching the data type
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# training the model
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=256)

model.summary()
#end
=======
#import the required libraries
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm



# open the .txt file which have names of training videos
f = open("trainlist01.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
train = pd.DataFrame()
train['video_name'] = videos
train = train[:-1]
train.head()



# open the .txt file which have names of test videos
f = open("testlist01.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
test = pd.DataFrame()
test['video_name'] = videos
test = test[:-1]
test.head()



# creating tags for training videos
train_video_tag = []
for i in range(train.shape[0]):
    train_video_tag.append(train['video_name'][i].split('/')[0])

train['tag'] = train_video_tag

# creating tags for test videos
test_video_tag = []
for i in range(test.shape[0]):
    test_video_tag.append(test['video_name'][i].split('/')[0])

test['tag'] = test_video_tag


# storing the frames from training videos
for i in tqdm(range(train.shape[0])):
    count = 0
    videoFile = train['video_name'][i]
    i = 0
    path = ""
    while(True):
        path += videoFile[i]
        i = i + 1
        if(videoFile[i] == '/'):
            break
    cap = cv2.VideoCapture('Videos/'+ path + '/'+ videoFile.split(' ')[0].split('/')[1])   # capturing the video from the given path
    #print(cap.grab())
    #path should be given accordingly
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder named train_1
            filename ='train_1/' + videoFile.split('/')[1].split(' ')[0] +"_frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()


# getting the names of all the images
images = glob("train_1/*.jpg")
train_image = []
train_class = []
for i in tqdm(range(len(images))):
    # creating the image name
    p1 = images[i].split('/')[0]
    p1 = p1[8:]
    train_image.append(p1)
    # creating the class of image
    i = 2
    p2 = ""
    while(True):
        if(p1[i] == '_'):
            break
        p2 = p2 + p1[i]
        i = i + 1
    train_class.append(p2)

# storing the images and their class in a dataframe
train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

# converting the dataframe into csv file
train_data.to_csv('train_new.csv',header=True, index=False)



#import libraries
import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras import backend as K




train = pd.read_csv('train_new.csv')
train.head()

# creating an empty list
train_image = []

# for loop to read and store frames
for i in tqdm(range(train.shape[0])):

    # loading the image and keeping the target size as (224,224,3)(size varies according to requirement)
    img = image.load_img('train_1/'+train['image'][i], target_size=(200,200,3))

    # converting it to array
    img = image.img_to_array(img)

    # normalizing the pixel value
    img = img/255

    # appending the image to the train_image list
    train_image.append(img)

# converting the list to numpy array
X = np.array(train_image)

# shape of the array
print(X.shape)

# separating the target
y = train['class']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)


# creating dummies of target variable for train and validation set
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

import tensorflow as tf

# creating the base model of pre-trained models
#base_model = VGG16(weights='imagenet', include_top=False,input_shape=(100,100,3))
base_model = tf.keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet',  input_shape=(200,200,3))
#base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',  input_shape=(200,200,3))
#base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3))


# extracting features for training frames
X_train = base_model.predict(X_train)
X_train.shape


# extracting features for validation frames
X_test = base_model.predict(X_test)
X_test.shape

print(X_train.shape[0])


# reshaping the training as well as validation frames in single dimension,acc to size of X_train after prediction
X_train = X_train.reshape(59075, 6*6*1024)
X_test = X_test.reshape(14769, 6*6*1024)

# normalizing the pixel values
max = X_train.max()
print(max)
X_train = X_train/max
X_test = X_test/max

print(X_train.shape)


#defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu',input_shape=(36864,)))#size filled according to X_train size after prediction
model.add(Dropout(0.5))                                       #Dropout to reduce overfitting
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(101, activation='softmax'))

# defining a function to save the weights of best model
from tensorflow.keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint('weight_200_200_dense.hdf5', save_best_only=True, monitor='val_loss', mode='min')


# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


y_train = y_train.to_numpy(copy=False)
y_test = y_test.to_numpy(copy=False)


#matching the data type
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# training the model
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=256)

model.summary()
#end
>>>>>>> 44ff0abc5c2a1b0c3ef0b65dbea879cd017debb0
