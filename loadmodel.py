from numpy import loadtxt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from os import listdir
import cv2
import numpy as np


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(250, 250, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()
# load model
model.load_weights('first_try.h5')
# summarize model.
#model.summary()
images = listdir(r"C:\Users\tony\Desktop\gun_dataset\test\\")
print("images: "+str(images))
for image in images:
    pic = cv2.imread('test\\'+image)
    pic = cv2.resize(pic,(250,250))
    pic = np.reshape(pic,[1,250,250,3])
    #print(type(pic))
    #print(str(pic))
    print(str(model.predict(pic, batch_size=1)))
    print(str(model.predict_proba(pic)))
