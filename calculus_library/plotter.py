import sys
sys.path.append('../')
from mlrefined_libraries import basics_library as baslib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from IPython.display import display, HTML
import copy

# a short function for plotting function and derivative values over a large range for input function g
def ad_derval_plot(MyTuple,g,**kwargs):
    # specify range of input for our function and its derivative
    w = np.linspace(-10,10,1000) 
    if 'w' in kwargs:
        w = kwargs['w']
  
    # recurse to create higher order derivative object
    order = 1
    if 'order' in kwargs:
        order = kwargs['order']
    
    # initialize objects
    valder_objs = []
    for u in w:
        # loop over and make deep object for higher order derivatives
        s = MyTuple(val = u)
        for i in range(order-1):
            s = MyTuple(val = s)
        valder_objs.append(s)

    # collect function and derivative values to plot
    results = [g(w) for w in valder_objs]
    
    # loop over and collect final derivative value
    g = []
    dgdw = []
    for r in results:
        val = r.val
        der = r.der
        for i in range(order-1):
            val = val.val
            der = der.der
        g.append(val)
        dgdw.append(der)

    # generate original function
    function_table = np.stack((w,g), axis=1) 

    # generate derivative function
    derivative_table = np.stack((w,dgdw), axis=1) 

    # use custom plotter to show both functions
    ylabel_2 = r'$\frac{\mathrm{d}^' + str(order) +  '}{\mathrm{d}w^' + str(order) +  '}g(w)$'
    if order == 1: # remove superscripts from label
        ylabel_2 = r'$\frac{\mathrm{d}}{\mathrm{d}w}g(w)$' 
    
    baslib.basics_plotter.double_plot(table1 = function_table, table2 = derivative_table,plot_type = 'continuous',xlabel = '$w$',ylabel_1 = '$g(w)$',ylabel_2 = ylabel_2,fontsize = 12)

# plotter for function and derivative equations
def derval_eq_plot(g,dgdw,**kwargs):
    # specify range of input for our function and its derivative
    w = np.linspace(-10,10,1000) 
    if 'w' in kwargs:
        w = kwargs['w']
        
    # make real function / derivative values
    g_vals = g(w)
    dgdw_vals = dgdw(w)

    # generate original function
    function_table = np.stack((w,g_vals), axis=1) 

    # generate derivative function
    derivative_table = np.stack((w,dgdw_vals), axis=1) 

    # use custom plotter to show both functions
    baslib.basics_plotter.double_plot(table1 = function_table, table2 = derivative_table,plot_type = 'continuous',xlabel = '$w$',ylabel_1 = '$g(w)$',ylabel_2 = r'$\frac{\mathrm{d}}{\mathrm{d}w}g(w)$',fontsize = 14)
    
    
# custom plot for spiffing up plot of a two mathematical functions
def double_2d_plot(func1,func2,**kwargs): 
    # get labeling arguments
    xlabel = '$w$'
    ylabel_1 = ''
    ylabel_2 = ''
    title1=  ''
    title2 = ''
    fontsize = 13
    color = 'r'
    w = np.linspace(-2,2,1000)
    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    if 'ylabel_1' in kwargs:
        ylabel_1 = kwargs['ylabel_1']
    if 'ylabel_2' in kwargs:
        ylabel_2 = kwargs['ylabel_2']
    if 'fontsize' in kwargs:
        fontsize = kwargs['fontsize']
    if 'title1' in kwargs:
        title1 = kwargs['title1']
    if 'title2' in kwargs:
        title2 = kwargs['title2']
    if 'w' in kwargs:
        w = kwargs['w']
    if 'color' in kwargs:
        color = kwargs['color']
        
    # determine vertical plotting limit
    f1 = func1(w)
    f2 = func2(w)
    ymax = max(max(f1),max(f2))
    ymin = min(min(f1),min(f2))
    ygap = (ymax - ymin)*0.2
    ymax += ygap
    ymin -= ygap
        
    # plot the functions 
    fig = plt.figure(figsize = (8,3))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122);    
    ax1.plot(w, f1, c=color, linewidth=2,zorder = 3)
    ax2.plot(w, f2, c=color, linewidth=2,zorder = 3)

    # plot x and y axes, and clean up
    ax1.set_xlabel(xlabel,fontsize = fontsize)
    ax1.set_ylabel(ylabel_1,fontsize = fontsize,rotation = 0,labelpad = 20)
    ax2.set_xlabel(xlabel,fontsize = fontsize)
    ax2.set_ylabel(ylabel_2,fontsize = fontsize,rotation = 0,labelpad = 20)
    ax1.set_title(title1[1:])
    ax2.set_title(title2[1:])
    ax1.set_ylim([ymin,ymax])
    ax2.set_ylim([ymin,ymax])
    
    ax1.grid(True, which='both'), ax2.grid(True, which='both')
    ax1.axhline(y=0, color='k', linewidth=1), ax2.axhline(y=0, color='k', linewidth=1)
    ax1.axvline(x=0, color='k', linewidth=1), ax2.axvline(x=0, color='k', linewidth=1)
    plt.show()
    
# custom plot for spiffing up plot of a two mathematical functions
def double_2d3d_plot(func1,func2,**kwargs): 
    # get labeling arguments
    xlabel = '$w$'
    ylabel_1 = ''
    ylabel_2 = ''
    title1=  ''
    title2 = ''
    fontsize = 15
    color = 'r'
    if 'fontsize' in kwargs:
        fontsize = kwargs['fontsize']
    if 'title1' in kwargs:
        title1 = kwargs['title1']
    if 'title2' in kwargs:
        title2 = kwargs['title2']
    if 'w' in kwargs:
        w = kwargs['w']
    if 'color' in kwargs:
        color = kwargs['color']
        
    # determine vertical plotting limit
    w = np.linspace(-2,2,500)
    xx,yy = np.meshgrid(w,w)
    xx.shape = (xx.size,1)
    yy.shape = (yy.size,1)
    w3d = np.concatenate((xx,yy),axis=1)
    f1 = func1(w)
    f2 = func2(w3d.T)
    xx.shape = (500,500)
    yy.shape = (500,500)
    f2.shape = (500,500)
        
    # plot the functions 
    fig = plt.figure(figsize = (8,4))
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
    ax1 = plt.subplot(gs[0]);
    ax2 = plt.subplot(gs[1],projection='3d'); 
    
    
    ax1.plot(w, f1, c=color, linewidth=2,zorder = 3)
    ax2.plot_surface(xx, yy, f2, alpha = 0.3,color = color,rstride=50, cstride=50,linewidth=2,edgecolor = 'k')
        
    # plot x and y axes, and clean up
    ax1.set_xlabel(xlabel,fontsize = fontsize)
    ax1.set_ylabel(ylabel_1,fontsize = fontsize,rotation = 0,labelpad = 20)
    ax2.set_xlabel(r'$w_1$',fontsize = fontsize,labelpad = 10)
    ax2.set_ylabel(r'$w_2$',fontsize = fontsize,rotation = 0,labelpad = 20)
    ax2.set_yticks(np.arange(min(w), max(w)+1, 1.0))
    ax1.set_title(title1[1:])
    ax2.set_title(title2[:],y=1.08)
    ax2.view_init(20,-60)
    
    ax1.grid(True, which='both'), ax2.grid(True, which='both')
    ax1.axhline(y=0, color='k', linewidth=1)
    ax1.axvline(x=0, color='k', linewidth=1)
    plt.show()
    
# custom plot for spiffing up plot of a two mathematical functions
def triple_3dsum_plot(func1,func2,**kwargs): 
    # get labeling arguments
    xlabel = '$w$'
    ylabel_1 = ''
    ylabel_2 = ''
    title1=  ''
    title2 = ''
    title3 = ''
    fontsize = 15
    if 'fontsize' in kwargs:
        fontsize = kwargs['fontsize']
    if 'title1' in kwargs:
        title1 = kwargs['title1']
    if 'title2' in kwargs:
        title2 = kwargs['title2']
    if 'title3' in kwargs:
        title3 = kwargs['title3']
        
    if 'w' in kwargs:
        w = kwargs['w']
        
    # determine vertical plotting limit
    w = np.linspace(-2,2,500)
    xx,yy = np.meshgrid(w,w)
    xx.shape = (xx.size,1)
    yy.shape = (yy.size,1)
    w3d = np.concatenate((xx,yy),axis=1)
    f1 = func1(w3d.T)
    f2 = func2(w3d.T)
    xx.shape = (500,500)
    yy.shape = (500,500)
    f1.shape = (500,500)
    f2.shape = (500,500)
        
    # plot the functions 
    fig = plt.figure(figsize = (15,4))
    plt.style.use('ggplot')
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
    ax1 = plt.subplot(gs[0],projection='3d');
    ax2 = plt.subplot(gs[1],projection='3d'); 
    ax3 = plt.subplot(gs[2],projection='3d'); 
   
    # plot surfaces
    ax1.plot_surface(xx, yy, f1, alpha = 0.3,color = 'r',rstride=50, cstride=50,linewidth=2,edgecolor = 'k')
    ax2.plot_surface(xx, yy, f2, alpha = 0.3,color = 'r',rstride=50, cstride=50,linewidth=2,edgecolor = 'k')
    ax3.plot_surface(xx, yy,f1 + f2, alpha = 0.3,color = 'r',rstride=50, cstride=50,linewidth=2,edgecolor = 'k')
        
    # plot x and y axes, and clean up
    ax1.set_xlabel(r'$w_1$',fontsize = fontsize,labelpad = 5)
    ax1.set_ylabel(r'$w_2$',fontsize = fontsize,rotation = 0,labelpad = 5)
    ax1.set_yticks(np.arange(min(w), max(w)+1, 1.0))
    ax1.set_title(title1[:],y=1.08)
    ax1.view_init(20,-60)
    
    ax2.set_xlabel(r'$w_1$',fontsize = fontsize,labelpad = 5)
    ax2.set_ylabel(r'$w_2$',fontsize = fontsize,rotation = 0,labelpad = 5)
    ax2.set_yticks(np.arange(min(w), max(w)+1, 1.0))
    ax2.set_title(title2[:],y=1.08)
    ax2.view_init(20,-60)
    
    ax3.set_xlabel(r'$w_1$',fontsize = fontsize,labelpad = 5)
    ax3.set_ylabel(r'$w_2$',fontsize = fontsize,rotation = 0,labelpad = 5)
    ax3.set_yticks(np.arange(min(w), max(w)+1, 1.0))
    ax3.set_title(title3[:],y=1.08)
    ax3.view_init(20,-60)
    
    plt.show()