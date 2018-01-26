# import custom JS animator
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only

# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
from autograd import grad as compute_grad   
from autograd import hessian as compute_hess
import autograd.numpy as np
import math
import time

class Visualizer:
    '''
    Animate runs of gradient descent and Newton's method, showing the correspnoding Taylor Series approximations as you go along.
    Run the algorithm first, and input the resulting weight history into this wrapper.
    ''' 

    ##### animate gradient descent method using single-input function #####
    def gradient_descent(self,g,w_hist,**kwargs):
        # compute gradient of input function
        grad = compute_grad(g)              # gradient of input function

        # decide on viewing range
        wmin = -3.1
        wmax = 3.1
        if 'wmin' in kwargs:            
            wmin = kwargs['wmin']
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
            
        # initialize figure
        fig = plt.figure(figsize = (9,4))
        artist = fig
        
        # remove whitespace from figure
        #fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        #fig.subplots_adjust(wspace=0.01,hspace=0.01)

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,4,1]) 

        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        ax = plt.subplot(gs[1]); 

        # generate function for plotting on each slide
        w_plot = np.linspace(wmin,wmax,200)
        g_plot = g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        width = 30
        
        # make color spectrum for points
        colorspec = self.make_colorspec(w_hist)
        
        # animation sub-function
        num_frames = 2*len(w_hist)+2
        print ('starting animation rendering...')
        def animate(t):
            ax.cla()
            k = math.floor((t+1)/float(2))
            
            # print rendering update            
            if np.mod(t+1,25) == 0:
                print ('rendering animation frame ' + str(t+1) + ' of ' + str(num_frames))
            if t == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # plot function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 2)                           # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = w_hist[0]
                g_val = g(w_val)
                ax.scatter(w_val,g_val,s = 90,c = colorspec[k],edgecolor = 'k',linewidth = 0.5*((1/(float(k) + 1)))**(0.4),zorder = 3,marker = 'X')            # evaluation on function
                ax.scatter(w_val,0,s = 90,facecolor = colorspec[k],edgecolor = 'k',linewidth = 0.5*((1/(float(k) + 1)))**(0.4), zorder = 3)
                
                # draw dashed line connecting w axis to point on cost function
                s = np.linspace(0,g_val)
                o = np.ones((len(s)))
                ax.plot(o*w_val,s,'k--',linewidth=1)

            # plot all input/output pairs generated by algorithm thus far
            if k > 0:
                # plot all points up to this point
                for j in range(min(k-1,len(w_hist))):  
                    w_val = w_hist[j]
                    g_val = g(w_val)
                    ax.scatter(w_val,g_val,s = 90,c = colorspec[j],edgecolor = 'k',linewidth = 0.5*((1/(float(j) + 1)))**(0.4),zorder = 3,marker = 'X')            # plot point of tangency
                    ax.scatter(w_val,0,s = 90,facecolor = colorspec[j],edgecolor = 'k',linewidth =  0.5*((1/(float(j) + 1)))**(0.4), zorder = 2)
                    
            # plot surrogate function and travel-to point
            if k > 0 and k < len(w_hist) + 1:          
                # grab historical weight, compute function and derivative evaluations
                w = w_hist[k-1]
                g_eval = g(w)
                grad_eval = float(grad(w))
            
                # determine width to plot the approximation -- so its length == width defined above
                div = float(1 + grad_eval**2)
                w1 = w - math.sqrt(width/div)
                w2 = w + math.sqrt(width/div)

                # use point-slope form of line to plot
                wrange = np.linspace(w1,w2, 100)
                h = g_eval + grad_eval*(wrange - w)

                # plot tangent line
                ax.plot(wrange,h,color = colorspec[k-1],linewidth = 2,zorder = 1)      # plot approx

                # plot tangent point
                ax.scatter(w,g_eval,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 3,marker = 'X')            # plot point of tangency
            
                # plot next point learned from surrogate
                if np.mod(t,2) == 0 and k < len(w_hist) -1:
                    # create next point information
                    w_zero = w_hist[k]
                    g_zero = g(w_zero)
                    h_zero = g_eval + grad_eval*(w_zero - w)

                    # draw dashed line connecting the three
                    vals = [0,h_zero,g_zero]
                    vals = np.sort(vals)

                    s = np.linspace(vals[0],vals[2])
                    o = np.ones((len(s)))
                    ax.plot(o*w_zero,s,'k--',linewidth=1)

                    # draw intersection at zero and associated point on cost function you hop back too
                    ax.scatter(w_zero,h_zero,s = 100,c = 'k', zorder = 3,marker = 'X')
                    ax.scatter(w_zero,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 3)
                    ax.scatter(w_zero,g_zero,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 3, marker = 'X')            # plot point of tangency
                 
            # fix viewing limits
            ax.set_xlim([wmin-0.1,wmax+0.1])
            ax.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            
            # place title
            ax.set_xlabel(r'$w$',fontsize = 14)
            ax.set_ylabel(r'$g(w)$',fontsize = 14,rotation = 0,labelpad = 25)

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)
    
    ###### animate the method ######
    def newtons_method(self,g,w_hist,**kwargs):
        # compute gradient and hessian of input
        grad = compute_grad(g)              # gradient of input function
        hess = compute_hess(g)           # hessian of input function
        
        # set viewing range
        wmax = 3
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
        wmin = -wmax
        if 'wmin' in kwargs:
            wmin = kwargs['wmin']
            
        # initialize figure
        fig = plt.figure(figsize = (9,4))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,4,1]) 

        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        ax = plt.subplot(gs[1]); 
        
        # generate function for plotting on each slide
        w_plot = np.linspace(wmin,wmax,1000)
        g_plot = g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        w_vals = np.linspace(-2.5,2.5,50)
        width = 1
        
        # make color spectrum for points
        colorspec = self.make_colorspec(w_hist)
        
        # animation sub-function
        print ('starting animation rendering...')
        num_frames = 2*len(w_hist)+2
        def animate(t):
            ax.cla()
            k = math.floor((t+1)/float(2))
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if t == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            # plot function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 1)                           # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = w_hist[0]
                g_val = g(w_val)
                ax.scatter(w_val,g_val,s = 100,c = colorspec[k],edgecolor = 'k',linewidth = 0.7, marker = 'X',zorder = 2)            # plot point of tangency
                ax.scatter(w_val,0,s = 100,c = colorspec[k],edgecolor = 'k',linewidth = 0.7, zorder = 2)
                # draw dashed line connecting w axis to point on cost function
                s = np.linspace(0,g_val)
                o = np.ones((len(s)))
                ax.plot(o*w_val,s,'k--',linewidth=1,zorder = 0)
                
            # plot all input/output pairs generated by algorithm thus far
            if k > 0:
                # plot all points up to this point
                for j in range(min(k-1,len(w_hist))):  
                    w_val = w_hist[j]
                    g_val = g(w_val)
                    ax.scatter(w_val,g_val,s = 90,c = colorspec[j],edgecolor = 'k',marker = 'X',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    ax.scatter(w_val,0,s = 90,facecolor = colorspec[j],edgecolor = 'k',linewidth = 0.7, zorder = 2)
                          
            # plot surrogate function and travel-to point
            if k > 0 and k < len(w_hist) + 1:          
                # grab historical weight, compute function and derivative evaluations    
                w_eval = w_hist[k-1]
                if type(w_eval) != float:
                    w_eval = float(w_eval)

                # plug in value into func and derivative
                g_eval = g(w_eval)
                g_grad_eval = grad(w_eval)
                g_hess_eval = hess(w_eval)

                # determine width of plotting area for second order approximator
                width = 0.5
                if g_hess_eval < 0:
                    width = - width

                # setup quadratic formula params
                a = 0.5*g_hess_eval
                b = g_grad_eval - 2*0.5*g_hess_eval*w_eval
                c = 0.5*g_hess_eval*w_eval**2 - g_grad_eval*w_eval - width

                # solve for zero points
                w1 = (-b + math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)
                w2 = (-b - math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)

                # compute second order approximation
                wrange = np.linspace(w1,w2, 100)
                h = g_eval + g_grad_eval*(wrange - w_eval) + 0.5*g_hess_eval*(wrange - w_eval)**2 

                # plot tangent curve
                ax.plot(wrange,h,color = colorspec[k-1],linewidth = 2,zorder = 2)      # plot approx

                # plot tangent point
                ax.scatter(w_eval,g_eval,s = 100,c = 'm',edgecolor = 'k', marker = 'X',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
                # plot next point learned from surrogate
                if np.mod(t,2) == 0:
                    # create next point information
                    w_zero = w_eval - g_grad_eval/(g_hess_eval + 10**-5)
                    g_zero = g(w_zero)
                    h_zero = g_eval + g_grad_eval*(w_zero - w_eval) + 0.5*g_hess_eval*(w_zero - w_eval)**2

                    # draw dashed line connecting the three
                    vals = [0,h_zero,g_zero]
                    vals = np.sort(vals)

                    s = np.linspace(vals[0],vals[2])
                    o = np.ones((len(s)))
                    ax.plot(o*w_zero,s,'k--',linewidth=1)

                    # draw intersection at zero and associated point on cost function you hop back too
                    ax.scatter(w_zero,h_zero,s = 100,c = 'b',linewidth=0.7, marker = 'X',edgecolor = 'k',zorder = 3)
                    ax.scatter(w_zero,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 3)
                    ax.scatter(w_zero,g_zero,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, marker = 'X',zorder = 3)            # plot point of tangency
            
            # fix viewing limits on panel
            ax.set_xlim([wmin,wmax])
            ax.set_ylim([min(-0.3,min(g_plot) - ggap),max(max(g_plot) + ggap,0.3)])
            
            # add horizontal axis
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            
            # label axes
            ax.set_xlabel(r'$w$',fontsize = 14)
            ax.set_ylabel(r'$g(w)$',fontsize = 14,rotation = 0,labelpad = 25)
            
            # set tickmarks
            ax.set_xticks(np.arange(round(wmin), round(wmax) + 1, 1.0))
            ax.set_yticks(np.arange(round(min(g_plot) - ggap), round(max(g_plot) + ggap) + 1, 1.0))

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

        return(anim)
    
    ### makes color spectrum for plotted run points - from green (start) to red (stop)
    def make_colorspec(self,w_hist):
        # make color range for path
        s = np.linspace(0,1,len(w_hist[:round(len(w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(w_hist[round(len(w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        colorspec = []
        colorspec = np.concatenate((s,np.flipud(s)),1)
        colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)
        return colorspec