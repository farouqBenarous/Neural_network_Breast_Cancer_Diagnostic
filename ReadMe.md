# Breast Cancer Diagnostics using neural network


## Overview

activation function : Sigmoid 

Optimazer  : RMS Optimazer based on gradient decent

number of inputs  :12 wich are the features of the cells  : 
'radius ', 'texture', 'perimeter', 'area', 'smoothness ', 'compactness ', 'concavity', 'concave  '' dimension', 'points', 'symmetry', 'fractal', 
number of hidden layers : 4 in the best cases 

number of output : 1 because its an binary clasification (Malignant begningn)

#####how  the prediction works ?
So first to to train my model I have too a dataset bunch of examples of Cells whather its benign of malignant and then 
train my model with it after I finished the training you get the Best optimal  parameters  to make the predictions 

##Here are some helpful links:
######Dataset details to understand how it had been collected 
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
 

## Framework
Tensorflow 

## Usage

Just run ``python3 demo.py`` to see the results:

To visulaze your Graph in tensorboard just run ``tensorboard --logdir=/tmp/mnist_tutorial/``
wich the path depends on what you have set in the code "Logdir"

