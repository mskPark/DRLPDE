import torch
import math
import numpy as np

from torchviz import make_dot

import matplotlib.pyplot as plt
import cv2

import matplotlib as mpl

mpl.rcParams['figure.dpi']= 300
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['font.size'] = 12

def speedcontourplot():
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[6.4, 5.0])

    plt.tight_layout(pad=0.75)

    contour = ax[0,0].contourf( x1, x2, u, levels, cmap=plt.cm.viridis)

    plt.cm.coolwarm

    ax[0,0].set_title(r'$\tilde{u}$')

    ax[0,0].tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False
    )

    colorbar0_param = fig.add_axes(
        [ax[0,0].get_position().x1 + 0.01,
        ax[0,0].get_position().y0,
        0.01,
        ax[0,0].get_position().height])
    colorbar0 = plt.colorbar(contour0, ticks=level_tick, cax = colorbar0_param)
    colorbar0.ax.tick_params(labelsize=10)

    ax[0,0].text(-1.675,
                0,
                r'$t=0$',
                fontsize=10)
    pass

def computational_graph(model):
    ## Example of using torch viz to show the computational graph

    x = torch.randn((29,2))

    make_dot(model(x), params = dict(model.named_parameters()))
    
    pass

def RK4(xt, u):
    ### Calculate x(t+1) from x' = u(x,t) using Runge-Kutta 4
    ### xt = (x,t)
    ### Used for plotting streamlines and making animations with tracer particles

    xt0 = xt

    h = 0.01

    k1 = u(xt)

    xt[:,:-1] = xt0[:,:-1] + h*k1/2 
    xt[:,-1] = xt0[:,-1] + h/2
    k2 = u(xt)

    xt[:,:-1] = xt0[:,:-1] + h*k2/2 
    k3 = u(xt)

    xt[:,:-1] = xt0[:,:-1] + h*k3
    xt[:,-1] = xt0[:,-1] + h
    k4 = u(xt)

    xt1 = xt0 + h/6 * (k1 + k2 + k3 + k4)
    return xt1