 # import autograd functionality to bulid function's properly for optimizers
import autograd.numpy as np

# import matplotlib functionality
import matplotlib.pyplot as plt
from matplotlib import gridspec


def visual_comparison(x,weights):
    '''
    Visually compare the results of several runs of PCA applied to two dimensional input and 
    two principal components
    '''
    # do weights
    weights = np.array(weights)
    num_runs = np.ndim(weights)
    
    # plot data
    fig = plt.figure(figsize = (10,4))
    gs = gridspec.GridSpec(1, num_runs) 
    for run in range(num_runs):
        # create subplot
        ax = plt.subplot(gs[run],aspect = 'equal'); 
        w_best = weights[run]

        # scatter data
        ax.scatter(x[0,:],x[1,:],c = 'k')
        
        # plot pc 1
        ax.arrow(0, 0, w_best[0,0], w_best[0,1], head_width=0.25, head_length=0.5, fc='k', ec='k',linewidth = 4)
        ax.arrow(0, 0, w_best[0,0], w_best[0,1], head_width=0.25, head_length=0.5, fc='lime', ec='lime',linewidth = 3)

        # plot pc 2
        ax.arrow(0, 0, w_best[1,0], w_best[1,1], head_width=0.25, head_length=0.5, fc='k', ec='k',linewidth = 4)
        ax.arrow(0, 0, w_best[1,0], w_best[1,1], head_width=0.25, head_length=0.5, fc='lime', ec='lime',linewidth = 3)
        
        # plot vertical / horizontal axes
        ax.axhline(linewidth=0.5, color='k',zorder = 0)
        ax.axvline(linewidth=0.5, color='k',zorder = 0)
        ax.set_title('run ' + str(run + 1),fontsize=16)
        ax.set_xlabel(r'$x_1$',fontsize = 16)
        ax.set_ylabel(r'$x_2$',fontsize = 16,rotation = 0,labelpad = 10)
        
def show_encode_decode(x,encoder,decoder,cost_history,weight_history,**kwargs):
    '''
    Examine the results of linear or nonlinear PCA / autoencoder to two-dimensional input.
    Four panels are shown: 
    - original data (top left panel)
    - data projected onto lower dimensional curve (top right panel)
    - lower dimensional curve (lower left panel)
    - vector field illustrating how points in space are projected onto lower dimensional curve (lower right panel)
    
    Inputs: 
    - x: data
    - encoder: encoding function from autoencoder
    - decoder: decoding function from autoencoder
    - cost_history/weight_history: from run of gradient descent minimizing PCA least squares
    
    Optinal inputs:
    - show_pc: show pcs?   Only useful really for linear case.
    - scale: for vector field / quiver plot, adjusts the length of arrows in vector field
    '''
    # user-adjustable args
    show_pc = False
    if 'show_pc' in kwargs:
        show_pc = kwargs['show_pc']
    scale = 14
    if 'scale' in kwargs:
        scale = kwargs['scale']
    encode_label = ''
    if 'encode_label' in kwargs:
        encode_label = kwargs['encode_label']

    # pluck out best weights
    ind = np.argmin(cost_history)
    w_best = weight_history[ind]

    ###### figure 1 - original data, encoded data, decoded data ######
    fig = plt.figure(figsize = (10,4))
    gs = gridspec.GridSpec(1, 3) 
    ax1 = plt.subplot(gs[0],aspect = 'equal'); 
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 
    ax3 = plt.subplot(gs[2],aspect = 'equal'); 

    # scatter original data with pc
    ax1.scatter(x[0,:],x[1,:],c = 'k',s = 60,linewidth = 0.75,edgecolor = 'w')

    if show_pc == True:
        for pc in range(np.shape(w_best)[1]):
            ax1.arrow(0, 0, w_best[0,pc], w_best[1,pc], head_width=0.25, head_length=0.5, fc='k', ec='k',linewidth = 4)
            ax1.arrow(0, 0, w_best[0,pc], w_best[1,pc], head_width=0.25, head_length=0.5, fc='lime', ec='lime',linewidth = 3)

    ### plot encoded and decoded data ###
    v = 0
    p = 0
    if np.ndim(w_best) > 2:
        # create encoded vectors
        v = encoder(w_best[0],x)

        # decode onto basis
        p = decoder(w_best[1:],v)
    else:
        # create encoded vectors
        v = encoder(w_best,x)

        # decode onto basis
        p = decoder(w_best,v)

    # plot decoded data 
    z = np.zeros((1,np.size(v)))
    ax2.scatter(v,z,c = 'k',s = 60,linewidth = 0.75,edgecolor = 'w')
    
    # plot decoded data 
    ax3.scatter(p[0,:],p[1,:],c = 'k',s = 60,linewidth = 0.75,edgecolor = 'w')

    # clean up panels
    xmin1 = np.min(x[0,:])
    xmax1 = np.max(x[0,:])
    xmin2 = np.min(x[1,:])
    xmax2 = np.max(x[1,:])
    xgap1 = (xmax1 - xmin1)*0.2
    xgap2 = (xmax2 - xmin2)*0.2
    xmin1 -= xgap1
    xmax1 += xgap1
    xmin2 -= xgap2
    xmax2 += xgap2
    
    for ax in [ax1,ax2,ax3]:
        if ax == ax1 or ax == ax3:
            ax.set_xlim([xmin1,xmax1])
            ax.set_ylim([xmin2,xmax2])
            ax.set_xlabel(r'$x_1$',fontsize = 16)
            ax.set_ylabel(r'$x_2$',fontsize = 16,rotation = 0,labelpad = 10)
            ax.axvline(linewidth=0.5, color='k',zorder = 0)
        else:
            ax.set_ylim([-1,1])
            if len(encode_label) > 0:
                ax.set_xlabel(encode_label,fontsize = 16)
        ax.axhline(linewidth=0.5, color='k',zorder = 0)
    
    ax1.set_title('original data',fontsize = 18)
    ax2.set_title('encoded data',fontsize = 18)
    ax3.set_title('decoded data',fontsize = 18)
    
    # set whitespace
    #fgs.update(wspace=0.01, hspace=0.5) # set the spacing between axes. 
        
    ##### bottom panels - plot subspace and quiver plot of projections ####
    fig = plt.figure(figsize = (10,4))
    gs = gridspec.GridSpec(1, 2) 
    ax1 = plt.subplot(gs[0],aspect = 'equal'); 
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 
    
    a = np.linspace(xmin1,xmax1,200)
    b = np.linspace(xmin2,xmax2,200)
    s,t = np.meshgrid(a,b)
    s.shape = (1,len(a)**2)
    t.shape = (1,len(b)**2)
    z = np.vstack((s,t))

    if np.ndim(w_best) > 2:
        # create encoded vectors
        v = encoder(w_best[0],z)

        # decode onto basis
        p = decoder(w_best[1:],v)
    else:
        # create encoded vectors
        v = encoder(w_best,z)

        # decode onto basis
        p = decoder(w_best,v)
    
    ax1.scatter(p[0,:],p[1,:],c = 'k',s = 1.5)
    ax2.scatter(p[0,:],p[1,:],c = 'k',s = 1.5)

    ### create quiver plot of how data is projected ###
    new_scale = 0.75
    a = np.linspace(xmin1 - xgap1*new_scale,xmax1 + xgap1*new_scale,20)
    b = np.linspace(xmin2 - xgap2*new_scale,xmax2 + xgap2*new_scale,20)
    s,t = np.meshgrid(a,b)
    s.shape = (1,len(a)**2)
    t.shape = (1,len(b)**2)
    z = np.vstack((s,t))

    if np.ndim(w_best) > 2:
        # create encoded vectors
        v = encoder(w_best[0],z)

        # decode onto basis
        p = decoder(w_best[1:],v)
    else:
        # create encoded vectors
        v = encoder(w_best,z)

        # decode onto basis
        p = decoder(w_best,v)

    # get directions
    d = []
    for i in range(p.shape[1]):
        dr = (p[:,i] - z[:,i])[:,np.newaxis]
        d.append(dr)
    d = 2*np.array(d)
    d = d[:,:,0].T
    M = np.hypot(d[0,:], d[1,:])
    ax2.quiver(z[0,:], z[1,:], d[0,:], d[1,:],M,alpha = 0.5,width = 0.01,scale = scale,cmap='autumn') 
    ax2.quiver(z[0,:], z[1,:], d[0,:], d[1,:],edgecolor = 'k',linewidth = 0.25,facecolor = 'None',width = 0.01,scale = scale) 

    #### clean up and label panels ####
    for ax in [ax1,ax2]:
        ax.set_xlim([xmin1 - xgap1*new_scale,xmax1 + xgap1*new_scale])
        ax.set_ylim([xmin2 - xgap2*new_scale,xmax2 + xgap1*new_scale])
        ax.set_xlabel(r'$x_1$',fontsize = 16)
        ax.set_ylabel(r'$x_2$',fontsize = 16,rotation = 0,labelpad = 10)
     
    ax1.set_title('manifold found',fontsize = 18)
    ax2.set_title('projection map',fontsize = 18)
    
    # set whitespace
    gs.update(wspace=0.01, hspace=0.5) # set the spacing between axes. 