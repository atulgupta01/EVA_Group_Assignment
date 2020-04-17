# EVA-Track3-Phase2-Session5
This repository contains assignment files of TSAI-EVA-Track3 course's Phase2 session1.

## Problem Statement
Design and train a CNN model on MNIST dataset so as to satisfy following conditions

   1) Model has less than 15k params
   
   2) Model uses dropout of 0.1 
   
   3) Model uses batchnorm
   
   4) Model uses randomrotate transform
   
   5) Model uses StepLR with step size = 6 and gamma = 0.1
   
   6) It achieves 99.3% test accuracy in less than 15 epochs
 

## Group Members

Atul Gupta (samatul@gmail.com)

Gaurav Patel (gaurav4664@gmail.com)

Ashutosh Panda (ashusai.panda@gmail.com)


## Files

**EVAP2S5_PyTorch.ipynb**

The Designed model has 14,848 parameters. Image random rotation is used for data augmentation. For model training stepLR strategy has been used. It is able to achieve test accuracy of 99.33% in 8 epoch itself. Colab file can also be found [here](https://colab.research.google.com/drive/13OW_xD-tpx5MVXBvPlYG4lxT3pE6mEAm).


