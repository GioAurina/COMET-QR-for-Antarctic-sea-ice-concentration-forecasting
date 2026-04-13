from functools import partial
import torch.utils.data as data

from torch.utils.tensorboard import SummaryWriter

import jax
from jax import jit

import numpy as onp

import optax

from tqdm.autonotebook import tqdm

@partial(jit, static_argnums = (0, 4))
def train_step(loss_fun, state, batch, 
               loss_mask = True, 
               loss_misfit_fn = optax.l2_loss,
               loss_static_args = ()):
    """
    jitted training step. 
    loss_fun: a loss function
    state: flax state
    batch: current batch
    loss_mask: mask values in the loss
    loss_misfit_fn: binary misfit fun
    loss_static_args: any additional args passed to loss (note: not jit-static!)
    """

    print('jitting train_step..', end = '')
    # Gradient function
    grad_fn = jax.value_and_grad(loss_fun)

    # Determine gradients for current model, parameters and batch
    loss, grads = grad_fn(state.params, state.apply_fn, batch, 
                          loss_mask, loss_misfit_fn, loss_static_args)

    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)

    # Return state and any other value we might want
    return state, loss

def train_model_val(loss_fun, 
                state, 
                data_loader: data.DataLoader, 
                val_loader: data.DataLoader, 
                jit_loss_fun, 
                loss_mask = True, 
                loss_misfit_fn = optax.l2_loss, 
                loss_static_args = (),
                writer: SummaryWriter = None, 
                num_step0 = 0, 
                num_epochs=100):
    
    """
    train with validation set, outputing states of min loss, min val loss and final state

    writer: torch.utils.tensorboard.SummaryWriter, which will log scalar loss
    log_step0: starting epoch for logging
    """
    
    Loss_train = onp.zeros((num_epochs, len(data_loader)))
    Loss_val = onp.zeros((num_epochs, len(val_loader)))

    loss_min_train = onp.inf
    loss_min_val = onp.inf

    state_min_train = None
    state_min_val = None

    epoch_min_train = 0
    epoch_min_val = 0

    is_print = True

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        
        # training epoch
        for i, batch in enumerate(data_loader):
            state, loss = train_step(loss_fun, state, batch, 
                                     loss_mask, loss_misfit_fn, loss_static_args)
            
            if is_print: 
                print('Done')
                is_print = False

            Loss_train[epoch, i]=loss

            if loss<loss_min_train:
                loss_min_train = loss
                state_min_train = state
                epoch_min_train = epoch

        # validation epoch
        for i, batch in enumerate(val_loader):
            loss = jit_loss_fun(state.params, state.apply_fn, batch, loss_mask, loss_misfit_fn, loss_static_args)
            
            Loss_val[epoch, i]=loss


            # save min val state
            if loss<loss_min_val:
                loss_min_val = loss
                state_min_val = state
                epoch_min_val = epoch

  
        # log to tensorboard
        if writer is not None:  
            writer.add_scalars('loss', 
                               {'train': Loss_train[epoch].mean(), 
                                'val': Loss_val[epoch].mean()}, 
                                global_step = epoch+num_step0)
            
    ckpt = {
        'train_state': state, 
        'state_min_train': state_min_train, 
        'state_min_val': state_min_val, 
        'Loss_train': Loss_train, 
        'Loss_val': Loss_val, 
        'loss_min_val': loss_min_val, 
        'epoch_min_val': epoch_min_val, 
        'loss_min_train': loss_min_train, 
        'epoch_min_train': epoch_min_train
        }    
    
    return ckpt


def train_model(loss_fun, 
                state, 
                data_loader: data.DataLoader, 
                loss_mask = True, 
                loss_misfit_fn = optax.l2_loss, 
                loss_static_args = (),
                writer: SummaryWriter = None, 
                num_step0 = 0, 
                num_epochs=100):
    
    """
    writer: torch.utils.tensorboard.SummaryWriter, which will log scalar loss
    log_step0: starting epoch for logging
    """
    
    Loss = onp.zeros((num_epochs, len(data_loader)))
    is_print = True

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for i, batch in enumerate(data_loader):
            
            state, loss = train_step(loss_fun, state, batch, 
                                     loss_mask, loss_misfit_fn, loss_static_args)
            
            if is_print: 
                print('Done') # signal finish of jitting
                is_print = False

            # We could use the loss and accuracy for logging here, e.g. in TensorBoard
            Loss[epoch, i]=loss

        if writer is not None:  
            writer.add_scalar('loss epoch', Loss[epoch].mean(), epoch+num_step0)
            
    return state, Loss
