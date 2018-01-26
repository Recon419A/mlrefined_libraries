# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib import gridspec

# import autograd functionality
import numpy as np
import math
import time
import copy

class Visualizer:
    '''
    Various plotting and visualization functions for illustrating training / fitting of nonlinear regression and classification
    '''             
    
    # compare regression cost histories from multiple runs
    def compare_regression_histories(self,histories,start,**kwargs):
        # initialize figure
        fig = plt.figure(figsize = (8,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        # any labels to add?        
        labels = [' ',' ']
        if 'labels' in kwargs:
            labels = kwargs['labels']

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(histories)):
            history = histories[c]
            
            label = 0
            if c == 0:
                label = labels[0]
            else:
                label = labels[1]
                
            # check if a label exists, if so add it to the plot
            if np.size(label) == 0:
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c)) 
            else:               
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),label = label) 

        # clean up panel
        ax.set_xlabel('iteration',fontsize = 12)
        ax.set_ylabel('cost function value',fontsize = 12)
        ax.set_title('cost function value at each step of gradient descent',fontsize = 15)
        if np.size(label) > 0:
            plt.legend(loc='upper right')
        ax.set_xlim([start - 1,len(history)+1])
        plt.show()
        
       
        
        
