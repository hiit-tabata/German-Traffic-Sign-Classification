# -*- coding: utf-8 -*-

# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle
from pathlib import Path
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import keras
import tensorflow as tf

import matplotlib.pyplot as plt



training_file = "./dataset/train.p"
validation_file="./dataset/valid.p"
testing_file = "./dataset/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train_original, y_train_original = train['features'], train['labels']
print("Number of training examples =", len(X_train_original))
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']



labels, counts = np.unique(y_train_original, return_counts=True)


# params
NEW_DATA_FILE_NAME = "./dataset/train_aug.p"
NB_NEW_IMAGES = 0 #12000
ORIGINAL_DATA_FILE_NAME = "./dataset/train.p" 
#augmentation params
ANGLE_ROTATE = 20
TRANSLATION = 0.2

def augmenteImage(image, angle, translation):
    h, w, c = image.shape
    
    # random rotate
    angle_rotate = np.random.uniform(-angle, angle)
    rotation_mat = cv2.getRotationMatrix2D((w//2, h//2), angle_rotate, 1)

    img = cv2.warpAffine(image, rotation_mat, (w, h))

    # random translation
    x_offset = translation * w * np.random.uniform(-1, 1)
    y_offset = translation * h * np.random.uniform(-1, 1)
    mat = np.array([[1, 0, x_offset], [0, 1, y_offset]])

    # return warpped img
    return cv2.warpAffine(img, mat, (w, h))


if not Path(NEW_DATA_FILE_NAME).is_file():
    with open(ORIGINAL_DATA_FILE_NAME, mode='rb') as file:
        data = pickle.load(file)
    data_x, data_y = data['features'], data['labels']

    new_data_x = np.empty((NB_NEW_IMAGES, 32, 32, 3), dtype=np.uint8)
    new_data_y = np.empty(NB_NEW_IMAGES, dtype=np.uint8)

    for i in range(NB_NEW_IMAGES):
        target = np.random.randint(data_x.shape[0])
        new_data_x[i] = augmenteImage(data_x[target], ANGLE_ROTATE, TRANSLATION)
        new_data_y[i] = data_y[target]

    X_train = np.concatenate((data_x, new_data_x))
    y_train = np.concatenate((data_y, new_data_y))
    
    new_data = {'features': X_train, 'labels': y_train}

    with open(NEW_DATA_FILE_NAME, mode='wb') as file:
        pickle.dump(new_data, file)
else:    
    with open(NEW_DATA_FILE_NAME, mode='rb') as f:
        train = pickle.load(f)
        X_train, y_train = train['features'], train['labels']

        
labels, counts = np.unique(y_train, return_counts=True)

# Plot the histogram
#plt.rcParams["figure.figsize"] = [15, 5]
#axes = plt.gca()
#axes.set_xlim([-1,43])

import numpy as np

# Number of training examples
n_train = len(X_train)

# Number of testing examples.
n_test = len(X_test)

# the shape of an traffic sign image
image_shape = X_train[0].shape

# number of unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# gray scale the images
def toGrayscale(rgb):
    result = np.zeros((len(rgb), 32, 32,1))
    result[...,0] = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])  
    return result

# normalize the images
def normalizeGrascale(grayScaleImages):
    return grayScaleImages/255

def processImages(rgbImages):
    return normalizeGrascale(toGrayscale(rgbImages))

def transformOnHot(nbClass, listClass):
    oneHot = np.zeros((len(listClass), nbClass))
    oneHot[np.arange(len(listClass)), listClass] = 1
    return oneHot

x_train_processed = processImages(X_train)
x_test_processed  = processImages(X_test)
x_valid_processed = processImages(X_valid)
y_train_processed = transformOnHot(n_classes, y_train)
y_valid_processed = transformOnHot(n_classes, y_valid)
y_test_processed  = transformOnHot(n_classes, y_test)



def showPicture():   
    assert(len(x_train_processed)  ==  len(X_train_original))
    
    f, axarr = plt.subplots(2, 1)
    plts = np.reshape(axarr, -1)
    
    
    for i in range(len(x_train_processed)):
        plts[0].imshow(x_train_processed[i][...,0], cmap=plt.cm.binary)
        plts[1].imshow(X_train_original[i])
        plt.waitforbuttonpress()




batch_size = 32
nb_classes = 10
nb_epoch = 20
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 1


model = Sequential()


with tf.name_scope('conv1'):
    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(32,32,1) ))
    model.add(Activation('relu'))
    
with tf.name_scope('conv2'):
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

with tf.name_scope('conv3'):
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    
with tf.name_scope('conv4'):
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

with tf.name_scope('fc1'):
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
with tf.name_scope('fc2'):
    model.add(Dense(43))
    model.add(Activation('softmax'))


tfBoard = keras.callbacks.TensorBoard(log_dir='./logs_1', histogram_freq=2, write_graph=True)
filepath="./weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


print('Not using data augmentation.')
model.fit(x_train_processed, y_train_processed,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(x_valid_processed, y_valid_processed),
          callbacks = [tfBoard, checkpoint],
          shuffle=True)



## test accc


print("predict test data")
y_res = tf.placeholder(tf.float32, [None, n_classes], name='y-input')
pred = model.predict_on_batch(x_test_processed)
acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y_res, 1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)   
    p = acc.eval({y_res:y_test_processed})
    print(p)









