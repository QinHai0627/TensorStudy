# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:53:18 2020

@author: QinHai
"""

import tensorflow as tf
import numpy as np
import keras
import gzip
import os

#import tensorflow.keras as keras
#import matplotlib.pyplot as plt
#from keras import models
#from keras import layers
#from keras.utils import to_categorical


 
#(train_images,train_labels),(test_images,test_labels)=keras.datasets.mnist.load_data()
#network = models.Sequential()
#network.add(layers.Dense(812,activation='relu',input_shape=(28*28,)))
#network.add(layers.Dense(10,activation='softmax'))
#network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#train_images = train_images.reshape((60000,28*28))
#train_images = train_images.astype('float32')/255
 
#test_images = test_images.reshape((10000,28*28))
#test_images = test_images.astype('float32')/255
 
#train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)
#network.fit(train_images,train_labels,epochs=5,batch_size=128)
#test_loss,test_acc = network.evaluate(test_images,test_labels)
#print('test_acc:',test_acc)

########################################

def load_data(data_folder):
    
    files=['train-labels-idx1-ubyte.gz', 
           'train-images-idx3-ubyte.gz',
           't10k-labels-idx1-ubyte.gz', 
           't10k-images-idx3-ubyte.gz']
    
    paths=[]
    
    for fname in files:
        paths.append(os.path.join(data_folder, fname))
        
    with gzip.open(paths[0],'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(),np.uint8,offset=8)
    
    with gzip.open(paths[1],'rb') as imgpath:
        train_images=np.frombuffer(imgpath.read(),np.uint8,offset=16).reshape(len(train_labels),28,28)

    with gzip.open(paths[2],'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(),np.uint8,offset=8)
        
    with gzip.open(paths[3],'rb') as imgpath:
        test_images=np.frombuffer(imgpath.read(),np.uint8,offset=16).reshape(len(test_labels),28,28)

    return (train_images,train_labels),(test_images,test_labels)

(train_images,train_labels), (test_images,test_labels) = load_data('C:/Users/QinHai/Downloads/Compressed')

#train_labels, test_labels = train_labels / 255.0, test_labels / 255.0

model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(812,activation='relu',input_shape=(28*28,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
    ])

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

train_images = train_images.reshape((60000,28*28))
#train_images = train_images.astype('float32')/255
 
test_images = test_images.reshape((10000,28*28))
#test_images = test_images.astype('float32')/255

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)
model.fit(train_images,train_labels,epochs=5,batch_size=16)
test_loss,test_acc = model.evaluate(test_images,test_labels)
print('test_acc:',test_acc)