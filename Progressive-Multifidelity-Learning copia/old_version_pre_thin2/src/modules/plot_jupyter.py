"""
Some handy plotting utils for use inside jupyter notebooks
"""

import numpy as onp

import matplotlib.pyplot as plt

from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from mpl_toolkits.axes_grid1 import make_axes_locatable

def contour_data(data, x = None, y = None, mask = True):
    """
    plot data of shape [N_data, x, y]
    """
    def plot(i, data, ind_x, ind_y, mask = True):
        
        state_i = onp.where(mask, data[i, ...], onp.nan)

        fig, ax = plt.subplots(ncols=3, nrows=1, figsize = (15, 5))
        # profile
        ax[0].set(adjustable='box', aspect=1) 
        im = ax[0].contourf(x, y, state_i, vmin = data_min, vmax = data_max)
        ax[0].axvline(x[ind_x])
        ax[0].axhline(y[ind_y], color = 'k')

        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax = cax, orientation='vertical')

        # y-slice
        # ax[1].set(adjustable='box', aspect=1) 
        ax[1].plot(x, state_i[ind_y, :], color = 'k')
        ax[1].set_xlim([x_min, x_max])
        ax[1].set_ylim([data_min, data_max])
        

        # x-slice
        # ax[2].set(adjustable='box', aspect=1) 
        ax[2].plot(y, state_i[:, ind_x])
        ax[2].set_xlim([y_min, y_max])
        ax[2].set_ylim([data_min, data_max])

        fig.tight_layout()
        

    if x is None: x = onp.arange(data.shape[1])
    if y is None: y = onp.arange(data.shape[2])

    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()

    data_min = data.min()
    data_max = data.max()
        


    i_widget = widgets.IntSlider(min=0, 
                                max=data.shape[0]-1, step=1, 
                                value=0, 
                                continuous_update=True, 
                                description = 'f_i(x,y)',
                                )
    
    ind_x_widget = widgets.IntSlider(min=0, 
                                max=data.shape[2]-1, step=1, 
                                continuous_update=True, 
                                value=data.shape[2]//2, 
                                description = 'f_i(x,:)',
                                )
    
    ind_y_widget = widgets.IntSlider(min=0, 
                                max=data.shape[1]-1, step=1, 
                                continuous_update=True, 
                                value=data.shape[1]//2, 
                                description = 'f_i(:, y)',
                                )

    w = interactive(plot, 
                    i = i_widget, 
                    ind_x = ind_x_widget, 
                    ind_y = ind_y_widget,
                    data = fixed(data), mask = fixed(mask))
    
    return w


def contour_compare(data1, data2, x = None, y = None, mask = True):
    """
    plot contourf of data1 and data2 of shapes [N_data, x, y] with widgets to choose i<N_data and slices

    returns widget w, which needs to be dispayed by running display(w)
    """
    def plot(i, BATCH, data2_BATCH, ind_x, ind_y, mask = True):
        
        state_i = onp.where(mask, BATCH[i, ...], onp.nan)
        data2_state_i = onp.where(mask, data2_BATCH[i, ...], onp.nan)


        fig, ax = plt.subplots(ncols=2, nrows=2, figsize = (5, 5))
        
        # profile 1
        ax[0, 0].set(adjustable='box', aspect=1) 
        ax[0, 0].contourf(x, y, state_i)
        ax[0, 0].axvline(x[ind_x], color = 'r', alpha = .5)
        ax[0, 0].axhline(y[ind_y], color = 'k', alpha = .5)
        ax[0, 0].set_xlim([x_min, x_max])
        ax[0, 0].set_ylim([y_min, y_max])
        
        # profile 2
        ax[0, 1].set(adjustable='box', aspect=1) 
        ax[0, 1].contourf(x, y, data2_state_i)
        ax[0, 1].axvline(x[ind_x], color = 'r')
        ax[0, 1].axhline(y[ind_y], color = 'k')
        ax[0, 1].set_xlim([x_min, x_max])
        ax[0, 1].set_ylim([y_min, y_max])

        # y-slice
        # ax[1].set(adjustable='box', aspect=1) 
        ax[1, 0].plot(x, state_i[ind_y, :], color = 'k', alpha = .5)
        ax[1, 0].plot(x, data2_state_i[ind_y, :], color = 'k', linestyle = '-')
        ax[1, 0].set_xlim([x_min, x_max])

        # x-slice
        # ax[2].set(adjustable='box', aspect=1) 
        ax[1, 1].plot(y, state_i[:, ind_x], color = 'r', alpha = .5)
        ax[1, 1].plot(y, data2_state_i[:, ind_x], color = 'r', linestyle = '-')
        ax[1, 1].set_xlim([y_min, y_max])

        
        

        fig.tight_layout()
        

    if x is None: x = onp.arange(data1.shape[2])
    if y is None: y = onp.arange(data1.shape[1])


    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()

    i_widget = widgets.IntSlider(min=0, 
                                max=data1.shape[0]-1, step=1, value=0, 
                                continuous_update=True, 
                                description = 'f_i(x,y)',
                                )
    
    ind_x_widget = widgets.IntSlider(min=0, 
                                max=data1.shape[2]-1, step=1, 
                                value=data1.shape[2]//2, 
                                continuous_update=True, 
                                description = 'f_i(x,:)',
                                )
    
    ind_y_widget = widgets.IntSlider(min=0, 
                                max=data1.shape[1]-1, step=1, 
                                value=data1.shape[1]//2,  
                                continuous_update=True, 
                                description = 'f_i(:, y)',
                                )

    w = interactive(plot, 
                    i = i_widget, 
                    ind_x = ind_x_widget, 
                    ind_y = ind_y_widget,
                    BATCH = fixed(data1), data2_BATCH = fixed(data2), mask = fixed(mask))
    
    return w


def plot_data1(data, x = None, fix_xlim = True, fix_ylim = False, mask = True):
    """
    slider plot of 1D data with time steps
    data has shape NtxNx or NbatchxNtxNx or list of arrays of shape NtxNx
      
    mask are indices to be filled with NaNs
    """

    def plotter(i, data, mask):
        state_i = onp.where(mask, data[:, i, :], onp.nan).T

        fig, ax = plt.subplots()
        ax.plot(x, state_i)

        if fix_xlim: ax.set_xlim([x_min, x_max])
        if fix_ylim: ax.set_ylim([data_min, data_max])

    
    

    # pass list of inputs of shape NtxNx
    if isinstance(data, list): data = onp.stack(data)
    elif data.ndim == 2: data = data[None, ...] # pass NtxNx
    

    if x is None: x = onp.arange(data.shape[-1])

    x_min = x.min()
    x_max = x.max()

    data_min = data.min()
    data_max = data.max()


    i_widget = widgets.IntSlider(min=0, 
                                max=data.shape[1]-1, step=1, 
                                value=0, 
                                continuous_update=True, 
                                description = 'f(i, x)',
                                )
    
    w = interactive(plotter, 
                    i = i_widget, 
                    data = fixed(data), mask = fixed(mask))
    
    return w