import keras
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPooling2D, UpSampling2D,Input,Activation,Add
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import numpy as np 
from numpy import genfromtxt
import matplotlib
#matplotlib.use('Agg')                            #uncomment for use on Euler
import matplotlib.pyplot as plt
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'          #to get rid of an error messsage in the ouput

#importing data 
my_data = genfromtxt('t_train_075100125.out')
t_train = np.reshape(my_data,(-1,100,100,1))
my_data = genfromtxt('t_val_075100125.out')
t_val = np.reshape(my_data,(-1,100,100,1))
my_data = genfromtxt('v_train_075100125.out')
v_train = np.reshape(my_data,(-1,100,100,1))
my_data = genfromtxt('v_val_075100125.out')
v_val = np.reshape(my_data,(-1,100,100,1))



input_shape=(100,100,1)
images = Input(input_shape)

#this function consists of 3 convolutional layers. 
def Conv_block(number_of_filters,kernel_size,x):

	x = Conv2D(number_of_filters,(kernel_size,kernel_size),activation='relu',padding='same')(x)
	x = Conv2D(number_of_filters,(kernel_size,kernel_size),activation='relu',padding='same')(x)
	x = Conv2D(number_of_filters,(kernel_size,kernel_size),activation='relu',padding='same')(x)
	return x

#the model architecture starts from here. The shape of layers will be given in between steps for clarity
x0 = Conv2D(24, (4, 4),activation='relu',padding='same',input_shape=(100,100,1))(images)
x0 = Conv2D(24,(4,4),activation='relu',padding='same')(x0)
x0 = Conv2D(24,(4,4),activation='relu',padding='same')(x0)
x0_shortcut = Conv_block(24,4,x0)

#shape = (100x100x24)

x= MaxPooling2D(pool_size=(2,2))(x0)
x1 = Conv_block(48,4,x)
x1_shortcut = Conv_block(48,4,x1)

# shape= (50x50x48)

x = MaxPooling2D(pool_size=(2,2))(x1)
x2 = Conv_block(96,4,x)
x2_shortcut = Conv_block(96,4,x2)

#shape = (25x25x96)

x = MaxPooling2D(pool_size=(5,5))(x2)
x3 = Conv_block(192,4,x)
x3_shortcut = Conv_block(192,4,x3)

#shape = (5x5x192)

x4 = MaxPooling2D(pool_size=(5,5))(x3)
x = Conv2D(384,(1,1),activation='relu')(x4)
x = Conv2D(384,(1,1),activation='relu')(x)

#shape =(1x1x384)

x = UpSampling2D(size=(5,5))(x)
x = Conv_block(192,4,x)

#shape = (5x5x192)

x = Add()([x, x3_shortcut])
x = UpSampling2D(size=(5,5))(x)
x = Conv_block(96,4,x)

#shape =  (25x25x96)

x = Add()([x, x2_shortcut])
x = UpSampling2D(size=(2,2))(x)
x = Conv_block(48,4,x)

#shape = (50x50x48)

x = Add()([x, x1_shortcut])
x = UpSampling2D(size=(2,2))(x)
x = Conv_block(24,4,x)

#shape = (100x100x24)

x = Add()([x, x0_shortcut])
x = Conv_block(12,4,x)
x = Conv2D(1,(4,4),activation='tanh',padding='same')(x)
#output

model = Model(input=images,output=x)

#specify loss function and optimizer
model.compile(loss='mean_absolute_error',optimizer=keras.optimizers.Adam(lr = 0.0000001 ))

#callback functions. These functions are used to stop training(call), save the current best model(checkpoint) and reduce the learning rate given some conditions(reducing)
call = EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=0, mode='min')
checkpoint = ModelCheckpoint('075100125_single.h5', monitor='val_loss', verbose=0, save_best_only=True,mode='min')
reducing = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, mode='min', epsilon=0, cooldown=0, min_lr=0)

#specify epochs,batch size 
history = model.fit(t_train, v_train,nb_epoch=1000000,batch_size=5,verbose=1,validation_data=(t_val,v_val),callbacks=[call,checkpoint,reducing])

#plotting the loss vs epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model error')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('075100125_loss.png')



