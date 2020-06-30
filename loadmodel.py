from numpy import loadtxt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from os import listdir
import cv2
import numpy as np


model = load_model('model.h5')

model.summary()
# load model
#model.load_weights('first_try.h5')
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
    print(image)
    print(str(model.predict(pic, batch_size=1)))
    #print(str(model.predict_proba(pic)))
