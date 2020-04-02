import numpy as np
from numpy import loadtxt,linspace
from random import randint
import matplotlib.pyplot as plt

#import data

t = loadtxt('t_val_087100112_double.out')
u = loadtxt('u_val_087100112_double.out')
v = loadtxt('v_val_087100112_double.out')


def build():    #building dataset out of other dataset
    t_1  = loadtxt('t_0.1.out')
    t_1 = np.reshape(t_1,(-1,10000))
    t_1_train = t_1[0::14,:]
    t_1_val = t_1[6::14,:]
    u_1 = loadtxt('u_0.1.out')
    u_1 = np.reshape(u_1,(-1,10000))
    u_1_train = u_1[0::14,:]
    u_1_val = u_1[6::14,:]
    v_1 = loadtxt('v_0.1.out')
    v_1 = np.reshape(v_1,(-1,10000))
    v_1_train = v_1[0::14,:]
    v_1_val = v_1[6::14,:]

    t_075  = loadtxt('t_0.075.out')
    t_075 = np.reshape(t_075,(-1,10000))
    t_075_train = t_075[0::14,:]
    t_075_val = t_075[6::14,:]
    u_075 = loadtxt('u_0.075.out')
    u_075 = np.reshape(u_075,(-1,10000))
    u_075_train = u_075[0::14,:]
    u_075_val = u_075[6::14,:]
    v_075 = loadtxt('v_0.075.out')
    v_075 = np.reshape(v_075,(-1,10000))
    v_075_train = v_075[0::14,:]
    v_075_val = v_075[6::14,:]

    t_125  = loadtxt('t_0.125.out')
    t_125 = np.reshape(t_125,(-1,10000))
    t_125_train = t_125[0::14,:]
    t_125_val = t_125[6::14,:]
    u_125 = loadtxt('u_0.125.out')
    u_125 = np.reshape(u_125,(-1,10000))
    u_125_train = u_125[0::14,:]
    u_125_val = u_125[6::14,:]
    v_125 = loadtxt('v_0.125.out')
    v_125 = np.reshape(v_125,(-1,10000))
    v_125_train = v_125[0::14,:]
    v_125_val = v_125[6::14,:]

    print(np.shape(t_075_train))
    print(np.shape(t_075_val))
    print(np.shape(u_075_train))
    print(np.shape(u_075_val))
    print(np.shape(v_075_train))
    print(np.shape(v_075_val))

    print(np.shape(t_125_train))
    print(np.shape(t_125_val))
    print(np.shape(u_125_train))
    print(np.shape(u_125_val))
    print(np.shape(v_125_train))
    print(np.shape(v_125_val))

    t_train=[]
    t_train.append(np.reshape(t_075_train,(-1,1)))
    t_train.append(np.reshape(t_1_train,(-1,1)))
    t_train.append(np.reshape(t_125_train,(-1,1)))
    t_train = np.asarray(t_train)
    t_train = np.reshape(t_train,(-1,1))

    t_val=[]
    t_val.append(np.reshape(t_075_val,(-1,1)))
    t_val.append(np.reshape(t_1_val,(-1,1)))
    t_val.append(np.reshape(t_125_val,(-1,1)))
    t_val = np.asarray(t_val)
    t_val = np.reshape(t_val,(-1,1))

    u_train=[]
    u_train.append(np.reshape(u_075_train,(-1,1)))
    u_train.append(np.reshape(u_1_train,(-1,1)))
    u_train.append(np.reshape(u_125_train,(-1,1)))
    u_train = np.asarray(u_train)
    u_train = np.reshape(u_train,(-1,1))

    u_val=[]
    u_val.append(np.reshape(u_075_val,(-1,1)))
    u_val.append(np.reshape(u_1_val,(-1,1)))
    u_val.append(np.reshape(u_125_val,(-1,1)))
    u_val = np.asarray(u_val)
    u_val = np.reshape(u_val,(-1,1))

    v_train=[]
    v_train.append(np.reshape(v_075_train,(-1,1)))
    v_train.append(np.reshape(v_1_train,(-1,1)))
    v_train.append(np.reshape(v_125_train,(-1,1)))
    v_train = np.asarray(v_train)
    v_train = np.reshape(v_train,(-1,1))

    v_val=[]
    v_val.append(np.reshape(v_075_val,(-1,1)))
    v_val.append(np.reshape(v_1_val,(-1,1)))
    v_val.append(np.reshape(v_125_val,(-1,1)))
    v_val = np.asarray(v_val)
    v_val = np.reshape(v_val,(-1,1))

    np.savetxt('t_train.out',t_train)
    np.savetxt('t_val.out',t_val)
    np.savetxt('u_train.out',u_train)
    np.savetxt('u_val.out',u_val)
    np.savetxt('v_train.out',v_train)
    np.savetxt('v_val.out',v_val)



def visualize():

  print("number of steps = ",len(t)/10000)
  a = input("line or pic ?")
  b = input("u,v or t ?")
  
  if b =='u':
    data = u
  elif b == 'v':
    data = v
  elif b == 't':
    data = t

  if a == 'pic' :
    
    i = int(input("start at which step ?"))    
    while i <len(data)/10000:
       print("showing ",i)
       plt.imshow(np.reshape(data[0+i*10000:(i+1)*10000],(100,100)))
       plt.show()
       i+=1
      
  if a =='line' :

   c = 'y'
   while c=='y':
       i = int(input("start at which step ?"))
       y = int(input("y coordinate ?(0,100)"))
       while i <len(data)/10000:
           print("showing ",i)
           plt.plot(np.reshape(data[0+i*10000:(i+1)*10000],(100,100))[y])
           plt.show()
           i+=1
       c = input("try again ?(y/n)")




def cut():

    my_data = loadtxt('t_raw.out')
    my_data = my_data[270*10000:330*10000]
    np.savetxt('t.out',my_data)
    my_data = loadtxt('u_raw.out')
    my_data = my_data[270*10000:330*10000]
    np.savetxt('u.out',my_data)
    my_data = loadtxt('v_raw.out')
    my_data = my_data[270*10000:330*10000]
    np.savetxt('v.out',my_data)


def analyze():

    y = int(input("y coordinate ?(0,100)"))
    v_max =[]
    i=0
    while i <len(v)/10000:
      v_max.append(float(max(np.reshape(v[0+i*10000:(i+1)*10000],(100,100))[y])))
      i+=1


    y_t = []
    i = 0
    while i <len(t)/10000:
        y = 0
        while y < 100:
         print("step",i,"y =",y)
         if float(max(np.reshape(t[0+i*10000:(i+1)*10000],(100,100))[y])) > 0.9 :
          y_top = y 
          print("found top :",y,"step :",i)
          while y < 100 :
            if float(max(np.reshape(t[0+i*10000:(i+1)*10000],(100,100))[y])) < 0.9 :
              y_bottom = y
              print("found bottom",y)
              y_t.append((y_top+y_bottom)/2)
              y = 100
            y += 1
         y += 1
        i += 1
    plt.figure(1)
    plt.plot(linspace(0,len(v)/10000,num=len(v)/10000),v_max)   
    plt.figure(2)
    plt.plot(linspace(0,len(t)/10000,num=len(t)/10000),y_t)
    plt.show()


def flip():  #flips images 
    
    i = 0
    t3 = np.reshape(t,(-1,100,100))
    u3 = np.reshape(u,(-1,100,100))
    v3 = np.reshape(v,(-1,100,100))
    t1 = np.reshape(t,(-1,100,100))
    u1 = np.reshape(u,(-1,100,100))
    v1 = np.reshape(v,(-1,100,100))
    t2 = []
    u2 = []
    v2 = []

    while i <np.ma.size(t1,0):

        t2.append(np.ravel(t3[i,:,:]))
        u2.append(np.ravel(u3[i,:,:]))
        v2.append(np.ravel(v3[i,:,:]))       

        t2.append(np.ravel(np.flipud(t1[i,:,:])))
        u2.append(np.ravel(np.flipud(u1[i,:,:])))
        v2.append(np.negative(np.ravel(np.flipud(v1[i,:,:]))))
        i += 1

    t2 = np.ravel(t2)
    u2 = np.ravel(u2)
    v2 = np.ravel(v2) 

    np.savetxt('t_val_087100112_double.out',t2)
    np.savetxt('u_val_087100112_double.out',u2)
    np.savetxt('v_val_087100112_double.out',v2)
    


def compare_flipped():

    v_normal = np.reshape(loadtxt('v_train.out'),(-1,100,100))
    v_flipped= np.reshape(loadtxt('v_test.out'),(-1,100,100))
    i = 0
    while i < 100 :
        plt.plot(v_normal[5,i,:])
        plt.plot(v_flipped[5,99-i,:])
        plt.show()
        i += 1



a = input("visualize,cut,analyze,build,flip,compare? ")

if a == 'visualize':
    visualize()
elif a =='build':
    build()
elif a =='cut':
    cut()
elif a =='analyze':
  analyze()
elif a == 'flip':
    flip()
elif a == 'compare' :
    compare_flipped()