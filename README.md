# bachelor_thesis

---------------------
CONTENTS OF THIS FILE
---------------------

 *Introduction
 *Folders
 *Simulation
 *Neural Network
 *Tools


INTRODUCTION
------------

Here you can find short explanations and instructions to run sample code from my Bachelor's Thesis.


FOLDERS
-------

Files are organized into 4 main folders.

Files for the fluid simulator are located in the 'simulation' folder.
Files for the neural network are located in the 'neural_network' folder.
Files for tools that I have used to manipulate data are in the 'tools' folder.
FIles for data are located in the 'data' folder.


SIMULATION
----------

Alongside with Gerris, gfsview2D is installed to visualize a simulation while it is running.
To run a sample simulation, simply run the following command in the directory /simulation in a linux terminal.

   sh run.sh

Simulation parameters are all contained in 'bubble.gfs'.


NEURAL NETWORK
--------------

Keras(2.0.8) is used with a TensorFlow(1.0.3) backend with python (3.6.6).

To train a neural network simply run the following command in a linux terminal.

   python neural_network.py

The file automatically takes sample data to train on.

To make a prediction for a velocity field using a trained model simply run the following command in the directory /neural_network in a linux terminal.

   python predict.py

It will use some samples for testing data .

A total of 6 trained models are located in neural_network/trained_models. A trained model file is structured like this.

   a_b.h5

a : 100 for 1 bubble only with size 100, 075100125 for 3 bubbles with sizes 075,100,125 ..etc.
b : this specifies whether the training set included bubbles in one or both sides.  Double for both sides and single for only one side.


DATA
----

All the data which is used for training and testing/validation can be found in the folder 'data'. Each file contains a 1-D vector of size 10000 multiplied by the number of data points. Each image is of resolution 100x100. A data file's name is structured like this.

   a_b_c.out

a : t for volume of fraction,v for vertical velocity, u for horizontal velocity
b : train for training set, val for validation set
c : 100 for 1 bubble only with size 100, 075100125 for 3 bubbles with sizes 075,100,125 ..etc.

To visualize some data run the following command in a linux terminal.

   python visualize.py

It will display the horizontal and vertical velocity fields and the volume fraction for one time-step at a time . 

TOOLS
-----

Some files that I have used to manipulate data can be found in 'tools'. These are not important but are included for completeness.

1) cut.py : this is used to cut, trim, analyze , visualize data from txt files
2) merge_vtk.py : this is used to convert vtk files into txt files .
3) merge_vtk_split.py : this is used to convert vtk files into txt files if we have parallel vtk files.
4) predict.py : this is used to make predictions and calculate the sample correlation coefficient
