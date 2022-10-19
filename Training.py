# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 06:14:33 2017

@author: Lutfi
"""

#KERAS

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam,adadelta,adamax,nadam,adagrad
from keras.utils import np_utils


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 132, 176

# number of channels
img_channels = 1

#%%
#  data

path1 = 'C:\\Users\\raisb\\trainingdata\\input_data'    #path of folder of images    
path2 = 'C:\\Users\\raisb\\trainingdata\\input_data_resized'  #path of folder to save images    

listing = os.listdir(path1) 
num_samples=size(listing)
print(num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)   
    img = im.resize((img_cols,img_rows))
    gray = img.convert('L')
                #need to do some more processing here           
    gray.save(path2 +'\\' +  file, "JPEG")

imlist = os.listdir(path2)

im1 = array(Image.open('C:\\Users\\raisb\\trainingdata\\input_data_resized' + '\\'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open('C:\\Users\\raisb\\trainingdata\\input_data_resized'+ '\\' + im2)).flatten()
              for im2 in imlist],'f')
                
label=np.ones((num_samples,),dtype = int)
label[0:800]=0 #candle
label[800:1600]=1 #noobs
label[1600:2399]=2 #obss

data,Label = shuffle(immatrix,label, random_state=2)
terserah = shuffle(immatrix)
train_data = [data,Label]

img=immatrix[110].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)


#%%
def plot_image(image):
    plt.imshow(image.reshape(img_rows,img_cols),
               interpolation='nearest',
               cmap='binary')

    plt.show()

#%%

image1 = data.x_test[0]
plot_image(image1)    
    
#%%
kernel ='gaussian'
#batch_size to train
batch_size = 10

# number of output classes
nb_classes = 6

# number of epochs to train
nb_epoch = 90

# number of convolutional filters to use
nb_filters = 16#32
nb_filters2 = 32
#nb_filters3 = 130

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_conv = 3

#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1)
X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1)

#tools for reading image
X_trains = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_tests = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
kernel = 'accuracy'
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#tools for reading labels
Y_trains = np_utils.to_categorical(y_train, nb_classes)
Y_tests = np_utils.to_categorical(y_test, nb_classes)


i = 10
plt.imshow(X_trains[i, 0], interpolation='nearest')
print("label : ", Y_trains[i,:])
#plt.imshow(X_train[i, 0],cmap='gray', interpolation='nearest')

#print (train_data[0].shape)
#print (train_data[1].shape)


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
model.add(Dense(8,init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


#adem = adadelta(lr=3)
model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=[kernel])
#model.compile(loss='mean_squared_error', optimizer=adem, metrics=[karnel])

#adem = adam(lr=0.0001)
#model.compile(optimizer=adem, loss='categorical_crossentropy', metrics=['accuracy'])

# Print Model Summary
#print(model.summary())

#SGD,RMSprop,adam,adadelta,adamax,nadam,adagrad

#load run untill here then run load code

#%%

#callbacks = TensorBoard(log_dir='./Graph')

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy = True, verbose=1, validation_data=(X_test, Y_test))
            
#hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#              show_accuracy = True, verbose=1, validation_split=0.2)


print(hist.history.keys())
# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
cek = val_acc
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train','Validation'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['seaborn-white'])

#print(plt.style.available)

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train Acc vs Validation Acc')
plt.grid(True)
plt.legend(['Train','Validation'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['seaborn-white'])





#%%       
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


print(model.predict_classes(X_test[0:5]))
print(Y_test[0:5])

i = 19
plt.imshow(X_test[i,0])
plt.imshow(img,cmap='gray')

#plt.imshow(X_tests[i, 0], interpolation='nearest')
print("label : ", Y_test[i,:])

#%%
# saving weights

fname = "Bismillah_Sukses_ambil_data_lengkap_n16_e60.hdf5"
model.save_weights(fname,overwrite=True)

# Loading weights

fname = "Bismillah_Sukses_ambil_data_lengkap_n16_e60.hdf5" 
model.load_weights(fname)
struktur=model.get_weights()


h=model.get_weights()
print (h)
tes = model.output_shape()
print (tes)
