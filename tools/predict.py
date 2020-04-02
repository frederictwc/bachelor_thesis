import keras
from keras.models import Sequential,load_model,Model
from keras.layers import Dense ,Conv2D,MaxPooling2D,Flatten, Dropout, advanced_activations, UpSampling2D,Input,BatchNormalization,Activation,merge,Conv2DTranspose,Add,concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.initializers import glorot_uniform
import numpy as np 
from numpy import genfromtxt
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
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
        
        model = load_model(input("which model? "))
        c = input("line or pic? ")
        a=0        
        #ground truth velocity
        while a <np.ma.size(x_val,0):
          if c == 'pic' :
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
          elif c == 'line' :
            
            while a <np.ma.size(x_val,0):

                y_t=[]                              #find center of bubble
                y = 0
                while y < 100:
                 if float(max(np.reshape(x_val,(-1,100,100))[a,y,:])) > 0.9 :
                  y_top = y 
                  while y < 100 :
                    if float(max(np.reshape(x_val,(-1,100,100))[a,y,:])) < 0.9 :
                      y_bottom = y
                      y_t = (int((y_top+y_bottom)/2))
                      y = 100
                    y += 1
                 y += 1

                b = x_val[a,:,:,:]
                b = np.array([b])
                y_testing = y_val[a,:,:,:]
                data_1 = model.predict(b) 
                plt.plot(np.reshape(y_testing,(100,100))[y_t,:])
                plt.plot(np.reshape(data_1,(100,100))[y_t,:])
                plt.legend(['ground truth', 'predicted'], loc='upper right')
                plt.xlabel('x')
                plt.ylabel('vertical velocity')
                plt.show()
                a+=1
        

def correlation(): #calculate correlation coefficient of validation samples

    a = 0
    image_correlation_coefficient = 0       #correlation coefficient calculated over whole image
    line_correlation_coefficient = 0        #correlation coefficient calculated on a horizontal axis through the center of the bubble
    dummy = 0
    model = load_model(input("which model? "))
    while a <np.ma.size(x_val,0):

        y_t=[]                              #find center of bubble
        y = 0
        while y < 100:
         if float(max(np.reshape(x_val,(-1,100,100))[a,y,:])) > 0.9 :
          y_top = y 
          while y < 100 :
            if float(max(np.reshape(x_val,(-1,100,100))[a,y,:])) < 0.9 :
              y_bottom = y
              y_t = (int((y_top+y_bottom)/2))
              y = 100
            y += 1
         y += 1

        b = x_val[a,:,:,:]
        b = np.array([b])
        y_testing = y_val[a,:,:,:]
        data_1 = model.predict(b) 
        print(y_t)
        plt.plot(np.reshape(y_testing,(100,100))[y_t,:])
        plt.plot(np.reshape(data_1,(100,100))[y_t,:])
        plt.show()
        line_correlation_coefficient  += np.corrcoef(np.reshape(data_1,(100,100))[y_t,:],np.reshape(y_testing,(100,100))[y_t,:])[1,0]
        image_correlation_coefficient += np.corrcoef(np.ravel(data_1),np.ravel(y_testing))[1,0]
        dummy                         += np.corrcoef(np.ravel(data_1),np.ones((10000))/5)[1,0]
        a+=1
     
    print("image correlation = ",image_correlation_coefficient/(a))
    print("line correlation = ",line_correlation_coefficient/(a))
    print("dummy correlation = ",dummy/(a))






a = input("predict or correlate?  ")
if a == "predict":
    predict()
elif a == "correlate":
    correlation()


