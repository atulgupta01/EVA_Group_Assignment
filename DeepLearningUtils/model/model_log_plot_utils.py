import numpy as np
import time, math
import tensorflow as tf

import matplotlib.pyplot as plt

def model_log_plot(trainAccuracy,testAccuracy,trainLosses,testLosses):
  fig, axs = plt.subplots(1,2,figsize=(15,5))   # define a plot with 1 row and 2 columns of subplots
      
  # summarize history for accuracy
  axs[0].plot(range(1,len(trainAccuracy)+1),trainAccuracy)
  axs[0].plot(range(1,len(testAccuracy)+1),testAccuracy)
  axs[0].set_title('Model Accuracy')
  axs[0].set_ylabel('Accuracy')
  axs[0].set_xlabel('Epoch')
  axs[0].set_xticks(np.arange(1,len(trainAccuracy)+1),len(trainAccuracy)/10)
  axs[0].legend(['train', 'val'], loc='best')

  # summarize history for loss
  axs[1].plot(range(1,len(trainLosses)+1),trainLosses)
  axs[1].plot(range(1,len(testLosses)+1),testLosses)
  axs[1].set_title('Model Loss') 
  axs[1].set_ylabel('Loss')
  axs[1].set_xlabel('Epoch')
  axs[1].set_xticks(np.arange(1,len(trainLosses)+1),len(trainLosses)/10)
  axs[1].legend(['train', 'val'], loc='best')
  plt.show()
