import torch
import time
from torch import nn
from typing import Optional
from ..models.models import BaselineModel, MultifidelityTransformer
import numpy as np
import pandas as pd
import os
from IPython.display import clear_output


class CustomMSE:
    def __init__(self):
        self.loss = nn.MSELoss(reduction="none")

    def __call__(self, y_hat, y):
        assert y_hat.size() == y.size()
        # Elementwise squared error
        squared_error = self.loss(
            y_hat.contiguous().view(-1, y_hat.size(-1)),
            y.contiguous().view(-1, y.size(-1)),
        )
        # Use nanmean to calculate the mean while ignoring NaN values
        return torch.nanmean(squared_error)
    

def lr_schedule(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def run_epoch(
    data_loader,
    model,
    loss_compute,
    optimizer,
    scheduler,
    log_refresh,
    mode="train",
    epoch_n=0,
    wandb_run=None,
    track_fidelity_losses=False,
):
    """Train a single epoch with automatic device handling."""
    start = time.time()
    total_loss = 0
    n_steps = 0
    
    # Set model mode explicitly
    if mode == "train":
        model.train()
    else:
        model.eval()
    
    # 1. DETECT DEVICE: Automatically find where the model is (e.g., 'mps')
    try:
        # Get the device of the first parameter of the model
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")
    
    # Initialize fidelity-level loss tracking
    fidelity_losses = {
        "3L_all": [], "2L_1+2": [], "2L_1+3": [], "2L_2+3": [],
        "1L_1": [], "1L_2": [], "1L_3": [],
    } if track_fidelity_losses else None
    
    # Wrap eval mode in no_grad context
    context_manager = torch.no_grad() if mode != "train" else torch.enable_grad()
    
    with context_manager:
        for step, batch in enumerate(data_loader):
            # 2. MOVE BATCH TO DEVICE: This fixes the RuntimeError
            # We iterate over the batch dict and move every tensor to the GPU/MPS
            batch = {
                key: val.to(device) if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }

            # Forward pass
            out = model.forward(batch)
            
            # Computation of the loss
            loss = loss_compute(out, batch["target"])
            
            # Track fidelity-level losses
            # Wrapped in no_grad for efficiency even in train mode (no backprop needed for tracking)
            if track_fidelity_losses and hasattr(model, '__class__') and 'Multifidelity' in model.__class__.__name__:
                with torch.no_grad():
                    masks = batch["mask"]
                    mask_patterns = {
                        "3L_all": [False, False, False],
                        "2L_1+2": [False, False, True],
                        "2L_1+3": [False, True, False],
                        "2L_2+3": [True, False, False],
                        "1L_1": [False, True, True],
                        "1L_2": [True, False, True],
                        "1L_3": [True, True, False],
                    }
                    
                    for pattern_name, pattern in mask_patterns.items():
                        pattern_tensor = torch.tensor(pattern, device=device)
                        mask_matches = (masks == pattern_tensor).all(dim=1)
                        
                        if mask_matches.any():
                            pattern_loss = loss_compute(out[mask_matches], batch["target"][mask_matches])
                            fidelity_losses[pattern_name].append(float(pattern_loss))
            
            if mode == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                if step % log_refresh == 1:
                    clear_output(wait=True)
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch: {epoch_n}\t||\tSteps: {n_steps}\t||\tLoss: {round(float(loss), 6)}\t||\tLearning Rate: {round(lr, 6)}\t||\tCompletion: {round(step / len(data_loader), 2)}"
                    )
                    if wandb_run is not None:
                        wandb_run.log({"learning_rate": lr})
                        
            total_loss += float(loss)
            n_steps += 1
            del loss
    
    final_time = time.time() - start
    
    # Calculate average fidelity losses
    if track_fidelity_losses:
        avg_fidelity_losses = {}
        for pattern_name, losses in fidelity_losses.items():
            if losses:
                avg_fidelity_losses[pattern_name] = sum(losses) / len(losses)
            else:
                avg_fidelity_losses[pattern_name] = None
        return total_loss / len(data_loader), final_time, avg_fidelity_losses
    
    return total_loss / len(data_loader), final_time

def model_eval(test_dataloader, scale_test, model, loss):
    # Here we assume that every test sample can be fetched with 
    # one pass (numerosity lower than batch size).
    test_samples = next(iter(test_dataloader))

    # Rescale predictions and targets to original scale.
    model.eval()
    with torch.no_grad():
        pred = model(test_samples).cpu() * scale_test
        output = test_samples["target"].cpu() * scale_test

    # Singular losses.
    single_losses = {}
    if isinstance(model, MultifidelityTransformer):
        single_losses = {id.item(): {} for id in test_samples["id"]}
        for sample_n, id in enumerate(test_samples["id"]):
            n_levels = (test_samples["mask"] == False)[sample_n].sum().item()
            curr_loss = round(loss(pred[sample_n, 1:], output[sample_n, 1:]).item(), 4)
            if f"Loss {n_levels}L" in single_losses[id.item()].keys():
                single_losses[id.item()][f"Loss {n_levels}L"].append(curr_loss)
            else:
                single_losses[id.item()][f"Loss {n_levels}L"] = [curr_loss]

        average_single_losses = {}
        for id, losses in single_losses.items():
            average_single_losses[id] = {}
            for level_name, loss_level in losses.items():
                average_single_losses[id][level_name] = round(np.array(loss_level).mean(), 4)

        single_losses = pd.DataFrame(average_single_losses).T
    else:
        for sample_n, id in enumerate(test_samples["id"]):
            curr_loss = round(loss(pred[sample_n, 1:], output[sample_n, 1:]).item(), 4)
            single_losses[id.item()] = curr_loss
        single_losses = pd.Series(single_losses, name="val_loss")
    
    return single_losses
