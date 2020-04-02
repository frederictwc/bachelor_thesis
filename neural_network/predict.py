import keras
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPooling2D, UpSampling2D,Input,Activation,merge,Add
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras import backend as K
import numpy as np 
from numpy import genfromtxt
import matplotlib
#matplotlib.use('Agg')                     #use if running on Euler 
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   

#import data

my_data = genfromtxt('t_train_075100125.out')
t_train = np.reshape(my_data,(-1,100,100,1))
my_data = genfromtxt('t_val_075100125.out')
t_val = np.reshape(my_data,(-1,100,100,1))
my_data = genfromtxt('v_train_075100125.out')
v_train = np.reshape(my_data,(-1,100,100,1))
my_data = genfromtxt('v_val_075100125.out')
v_val = np.reshape(my_data,(-1,100,100,1))


x_train = t_train
x_val  = t_val
y_train = v_train
y_val = v_val


def predict(): # Plot what a model predicts
        
        model = load_model('075100125_single.h5')
        a=0        

        while a <np.ma.size(x_val,0):

            b = x_val[a,:,:,:]
            b = np.array([b])
            y_testing = y_val[a,:]
            data_1 = model.predict(b)

            plt.figure(1)
            plt.imshow(np.reshape(y_testing,(100,100)),vmin=np.min(y_testing),vmax=np.max(y_testing))
            data_1 = model.predict(b)
            plt.figure(2)
            plt.imshow(np.reshape(data_1,(100,100)),vmin=np.min(y_testing),vmax=np.max(y_testing))
            plt.show()
            a+=1

      

predict()