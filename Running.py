# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:46:37 2018

@author: Lutfi
"""
#KERAS
import cv2 #need
import numpy as np #need
from PIL import Image
from numpy import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam,adadelta,adamax,nadam,adagrad
from keras.utils import np_utils

# input image dimensions
img_rows, img_cols = 144, 176

# number of channels
img_channels = 1

import time
import serial

ser=serial.Serial(
port='COM46',
baudrate=9600,
parity=serial.PARITY_NONE,
stopbits=serial.STOPBITS_ONE,
bytesize=serial.EIGHTBITS,
timeout=1
)

#%%
# input image dimensions
kernel ='gaussian'
#batch_size to train
batch_size = 10

# number of output classes
nb_classes = 3

# number of epochs to train
nb_epoch = 60

# number of convolutional filters to use
nb_filters = 16#32
nb_filters2 = 32
#nb_filters3 = 130

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_conv = 5

#%%
model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(img_cols, img_rows, 1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))


model.add(Convolution2D(nb_filters2, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))


#model.add(Convolution2D(nb_filters3, nb_conv, nb_conv))
#convout3 = Activation('relu')
#model.add(convout3)
#model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))


model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(16,init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


#adem = adadelta(lr=3)
#model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=[kernel])
#model.compile(loss='mean_squared_error', optimizer=adem, metrics=[karnel])
#adem = adadelta(lr=3)

# Compile model

#model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=[kernel])

fname = "Bismillah_Sukses_ambil_data_lengkap_n16_e60.hdf5"
model.load_weights(fname)

struktur=model.get_weights()

#%%
cap=cv2.VideoCapture(0)
cap.set(3,img_rows)
cap.set(4,img_cols)
ob=''
count = 0
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #F:\TUGAS AKHIR\PROGRAM\BISMILLAH CNN\test
    cv2.imwrite('F://TUGAS AKHIR//PROGRAM//BISMILLAH CNN//test//01.jpg',gray)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        count=count+1
        cv2.imwrite('F://TUGAS AKHIR//PROGRAM//BISMILLAH CNN//test//'+str(count)+'.jpg',frame)
    # Display the resulting frame
    
    path = 'F:\\TUGAS AKHIR\PROGRAM\\BISMILLAH CNN\\test\\01.jpg'
    img = Image.open(path)
    coba = img.resize((img_cols,img_rows))
    
    immatrix3=array([array(coba)],'f')
    lock=immatrix3.reshape(1,img_cols,img_rows,1)
    proses = lock
    Y_pred = model.predict(proses)
    boneka = Y_pred[0,0]
    hidup = Y_pred[0,1]
    no = Y_pred[0,2]

    if boneka > hidup and boneka > no: 
        ob="candle" 
        ser.write("3")
    elif hidup > boneka and hidup > no:
        ob="noobs"
        ser.write("2")
    elif no > boneka and no > hidup:
        ob="obs"    
        ser.write("1")
    font = cv2.FONT_HERSHEY_SIMPLEX  
    obs=ob
    cv2.putText(frame,obs, (5,20), font, 1, (255,255,255), 2 ,cv2.LINE_AA);
    cv2.imshow('Camera',frame)
    
    print(Y_pred)
    print(" result:",obs)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()