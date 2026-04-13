"""Shared imports and project path setup for the multifidelity GPU pipeline."""

import os
import sys
import gc
import argparse
import pickle
import dill
from pathlib import Path
from collections import Counter
from datetime import datetime

# Scientific computing
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

# PyDMD
from pydmd import DMD, BOPDMD, FbDMD, MrDMD
from pydmd.plotter import plot_eigs, plot_summary, plot_modes_2D
from pydmd.preprocessing import hankel_preprocessing

# Configuration
import yaml
import wandb
import warnings
warnings.filterwarnings("ignore")

# Add modules to path
sys.path.append('./src/modules')
from data_wrangle import thin_data, del_leap, get_days_before, window_mean, get_test_set, day_to_date, date_to_day
from dmd_routines import reshape_data2dmd, train_dmd, reshape_Psi2data, eval_dmd, eval_dmd_latent, bootstrap_train_dmd, eval_dmd_ensemble

# Import multifidelity modules
sys.path.append(os.getcwd())
sys.path.append("/home/u10715220")
from multifidelity_transformer.utils.data import MultiFidelityDataset, compute_randomized_SVD
from multifidelity_transformer.models.models_experimental import MultifidelityTransformer
from multifidelity_transformer.utils.training import lr_schedule, run_epoch, model_eval, CustomMSE
