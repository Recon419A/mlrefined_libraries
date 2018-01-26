# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from matplotlib import gridspec
import autograd.numpy as np
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only
from . import optimimzers
import copy
import time
import bisect


class Visualizer:
    '''
    Visualizer for stumps (depth 1 trees) for a N = 1 dimension input dataset
    '''

    # load target function
    def load_data(self,csvname):
        data = np.loadtxt(csvname,delimiter = ',')
        self.x = data[:,0]
        self.y = data[:,1]
        self.y.shape = (len(self.y),1)
        
    # initialize after animation call
    def dial_settings(self):
        self.colors = [[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5],'mediumaquamarine']
        
        #### initialize split points for trees ####
        # sort data by values of input
        self.x_t = copy.deepcopy(self.x)
        self.y_t = copy.deepcopy(self.y)
        sorted_inds = np.argsort(self.x_t,axis = 0)
        self.x_t = self.x_t[sorted_inds]
        self.y_t = self.y_t[sorted_inds]

        # create temp copy of data, sort from smallest to largest
        splits = []
        levels = []
        residual = copy.deepcopy(self.y_t)

        ## create simple 'weak learner' between each consecutive pair of points ##
        for p in range(len(self.x_t) - 1):
            if self.y_t[p] != self.y_t[p+1]:
                # determine points on each side of split
                split = (self.x_t[p] + self.x_t[p+1])/float(2)
                splits.append(split)

                # gather points to left and right of split
                pts_left  = [t for t in self.x_t if t <= split]
                resid_left = residual[:len(pts_left)]
                resid_right = residual[len(pts_left):]

                # compute average on each side
                ave_left = np.mean(resid_left)
                ave_right = np.mean(resid_right)
                levels.append([ave_left,ave_right])
                
        # randomize splits for this experiment
        self.splits = splits
        self.levels = levels
       
    ##### prediction functions #####
    # tree prediction
    def tree_predict(self,pt,w): 
        # our return prediction
        val = 0

        # loop over current stumps and collect weighted evaluation
        for i in range(len(self.splits)):
            # get current stump
            split = self.splits[i]
            levels = self.levels[i]
                
            # check - which side of this split does the pt lie?
            if pt <= split:  # lies to the left - so evaluate at left level
                val += w[i]*levels[0]
            else:
                val += w[i]*levels[1]
        return val

    ###### fit polynomials ######
    def browse_stumps(self,**kwargs):
        # set dials for tanh network and trees
        self.dial_settings()
        self.num_elements = len(self.splits)

        # construct figure
        fig = plt.figure(figsize = (9,5))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,3, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax = plt.subplot(gs[1]); ax.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # set viewing range for all 3 panels
        xmax = max(copy.deepcopy(self.x))
        xmin = min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.05
        xmax += xgap
        xmin -= xgap
        ymax = max(copy.deepcopy(self.y))[0]
        ymin = min(copy.deepcopy(self.y))[0]
        ygap = (ymax - ymin)*0.4
        ymax += ygap
        ymin -= ygap
        
        # animate
        print ('beginning animation rendering...')
        def animate(k):
            # clear the panel
            ax.cla()
            
            # print rendering update
            if np.mod(k+1,5) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(self.num_elements))
            if k == self.num_elements - 1:
                print ('animation rendering complete!')
                time.sleep(1)
                clear_output()
                                
            ####### plot a stump ######
            # pick a stump
            w = np.zeros((self.num_elements,1))
            w[k] = 1
            
            # produce learned predictor
            s = np.linspace(xmin,xmax,400)
            t = [self.tree_predict(np.asarray([v]),w) for v in s]

            # plot approximation and data in panel
            ax.scatter(self.x,self.y,c = 'k',edgecolor = 'w',s = 50,zorder = 2)
            ax.plot(s,t,linewidth = 2.5,color = self.colors[0],zorder = 3)
            
            # plot horizontal axis and dashed line to split point
            ax.axhline(c = 'k',linewidth = 1 ,zorder = 0) 
            mid = (self.levels[k][0] + self.levels[k][1])/float(2)
            o = np.linspace(0,ymax,100)
            e = np.ones((100,1))
            sp = self.splits[k]
            ax.plot(sp*e,o,linewidth = 1.5,color = self.colors[1], linestyle = '--',zorder = 1)
                
            # cleanup panel
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            ax.set_xlabel(r'$x$', fontsize = 14,labelpad = 10)
            ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 10)
            ax.set_xticks(np.arange(round(xmin), round(xmax)+1, 1.0))
            ax.set_yticks(np.arange(round(ymin), round(ymax)+1, 1.0))
            
        anim = animation.FuncAnimation(fig, animate,frames = self.num_elements, interval = self.num_elements, blit=True)
        
        return(anim)

 

 