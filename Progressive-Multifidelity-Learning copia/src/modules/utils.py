"""
Various genral utils
"""

import jax
import jax.numpy as np
from jax import vmap, random
from functools import partial
import optax
import numpy as onp
from tqdm.autonotebook import tqdm
from jax.tree_util import tree_map, tree_reduce

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.widgets import Button, Slider


from typing import Callable

# flatten params pytree into one vector for tensorboard
nn_flatten_onp = lambda x: onp.array(jax.tree_util.tree_reduce(
    lambda x,y: jax.numpy.concatenate((x.flatten(),y.flatten())),
                                                 x))

# compute difference between two trees, both with the same tree_structure
tree_delta = lambda tree1, tree2: tree_reduce(lambda x,y: np.linalg.norm(x)+np.linalg.norm(y), 
                                              tree_map(lambda x, y:x-y, tree1, tree2))


def gridPlot(a, nrows_ncols=None):
    if nrows_ncols is None:
        nrows_ncols = (1, len(a))
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols,  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    # plot oneach grid cell
    for ax, im in zip(grid, a):
        ax.imshow(im)

def slid_plot(Y_list, 
              x = None,
              T = None, 
              label_list = None, 
              ln_style_list = None,
              ind = None):
    """
    plot PDE solution with 1D output Y(x,t) with t set by slider
    Y_list: list of Y(x,t) or a single Y(x,t). 
            Each Y(x,t) is an Nt x Nx matrix. 
            They will be plotted on the same axis. 
    x: x-mesh
    T: t-mesh
    ind: number to be displayed in title (e.g., index of solution)

    Note:
        Run with cell magic:
        %matplotlib widget

    Example:
        # DATA2_val is of shape [10, 2001, 300] for 2001 time steps and 300 mesh points
        slid_plot([DATA2_val[0], DATA2_val[1], DATA2_val[2]], X)
    """
    
    plt.close('all')
    if type(Y_list) is not list:
        Y_list = [Y_list]

    N = len(Y_list)
    if label_list is None:
        if N==1:
            label_list = ['Truth']
            ln_style_list = ['-']
        
        elif N==2:
            label_list = ['Truth', 'ML']
            ln_style_list = ['-', '--']

        else:
            label_list = ['Truth']+[f'ML_{i}' for i in range(1, N)]
            ln_style_list = ['-']+[f'--' for i in range(1, N)]
            

    Nt = Y_list[0].shape[0]
    if T is None:
        T = onp.arange(Nt)

    Nx = Y_list[0].shape[1]
    if x is None:
        x = onp.arange(Nx)

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()


    y_ax_max = -onp.infty
    y_ax_min = onp.infty
    lines_list = []

    
    for i in range(N):
        line, = ax.plot(x, Y_list[i][0], 
                        linestyle = ln_style_list[i], 
                        label = label_list[i])
        lines_list.append(line)
        y_ax_max = max([y_ax_max, Y_list[i].max()])
        y_ax_min = min([y_ax_min, Y_list[i].min()])

    
    ax.set_ylim([y_ax_min, y_ax_max])
    title = ax.set_title(f'ind = {ind}, t={T[0]:.2f}')
    plt.legend(loc = 'upper right')

    
    ax_slider = fig.add_axes([0.15, .9, .7, .1 ])
    # slider = Slider(fig.add_subplot(50,1,48), '', valmin=0, valmax=Nt-1,  valstep=1)
    slider = Slider(ax_slider, '', valmin=0, valmax=Nt-1,  valstep=1)

    # Function to be called when slider value is changed
    def update(val):
        i = slider.val
        for k in range(N):
            lines_list[k].set_ydata(Y_list[k][i])
        title.set_text(f'ind = {ind}, t= {T[i]:.2f}, i = {i}')



    # Call update function when slider value is changed
    slider.on_changed(update);

    # resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    # button = Button(resetax, 'close fig', hovercolor='0.975')
    # def reset(event):
    #     plt.close('fig')
    # button.on_clicked(reset);


# custom init function
def normal(stddev=1e-2, dtype = np.float32) -> Callable:
  def init(key, shape, dtype=dtype):
    keys = random.split(key)
    return random.normal(keys[0], shape) * stddev
  return init

# compute covariance matrix for xs, xs2
@partial(jax.jit, static_argnums=(0))
def cov_map(cov_func, xs, xs2 = None):
    # xs and xs2 are stacked along the leading dimension
    if xs2 is None:
        return vmap(lambda x:  vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T


# This collate function is taken from the JAX tutorial with PyTorch Data Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
# for dataloader
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
