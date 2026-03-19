"""
Multifidelity Transformer - Residual Validation and Calibration (Ice Thickness & SST Version)
GPU-optimized version for external execution

This script converts the MF_Res_Val_Cal_Thickness_SST.ipynb notebook into an executable format
that can run on external GPU infrastructure and save all results to disk.

FIDELITY LEVELS:
    - Level 1 (x1): 64 POD modes of Ice Thickness (432×432 grid reduced via POD)
    - Level 2 (x2): 64 POD modes of Sea Surface Temperature (432×432 grid reduced via POD)
    - Level 3 (x3): Sensor measurements (128 sensors placed at high ice probability locations)

POD DIMENSION REDUCTION (FOR COMPUTATIONAL EFFICIENCY):
    - x1 and x2 use FULL 432x432 spatial grid (thin=1, no downsampling)
    - POD (Proper Orthogonal Decomposition) reduces 432x432=186,624 pixels → 64 modes
    - POD fitted ONLY on training data (no leakage to val/test)
    - Raw POD projections used (no singular value normalization)
    - ALL levels (x1, x2, x3) normalized with RobustScaler for balanced learning
    - mask_ice (static ice mask) used for target variable (y) and all computations
    - Speedup: ~2900x dimension reduction while preserving >95% energy

PREDICTION RECONSTRUCTION:
    - Final predictions = DMD baseline + learned residual quantiles
    - ALL predictions clipped to [0, 1] to maintain probability constraints
    - Clipping applied to: median, lower bound (5%), upper bound (95%), and ground truth

This script converts the notebook into an executable format

QUICK START:
    1. Test installation:  ./test_gpu_script.sh
    2. Run script:  python mf_res_val_cal_gpu.py --base_path /path/to/project
    3. Check results:  ls Results/ice3/

USAGE:
    python mf_res_val_cal_gpu.py [OPTIONS]

OPTIONS:
    --base_path PATH          Project directory path
    --experiment_name NAME    Experiment name (default: ice)
    --experiment_number NUM   Experiment number (default: 1)
    --device DEVICE           Device: auto/cuda/mps/cpu (default: auto)
    --no_wandb                Disable Weights & Biases logging
    --skip_training           Load existing model instead of training

EXAMPLES:
    # Basic run
    python mf_res_val_cal_gpu.py

    # Force CUDA and skip WandB
    python mf_res_val_cal_gpu.py --device cuda --no_wandb

    # Load existing model
    python mf_res_val_cal_gpu.py --skip_training

OUTPUTS:
    Results saved to: Results/<experiment_name>_<number>/
    - training_curves.png           Loss evolution plots
    - test_results.csv              Test metrics (CSV)
    - test_results.txt              Test metrics (formatted)
    - predictions_year*_day*.png    Sample predictions
    - all_predictions.pkl           Complete numerical data

REQUIREMENTS:
    - Data: 
        * data/ice/Antarctic_years_1989_2024i.pkl (ice concentration)
        * data/ice/Antarctic_thickness_1993_2023.pkl (ice thickness)
        * data/ice/Antarctic_SST_1993_2023.pkl (sea surface temperature)
    - DMD baseline: checkpoints/dmd_fits_all_years/dmd_forecasts_*.pkl
    - Config: multifidelity_transformer/experiment_configurations/ice/3.yaml
    - Packages: torch, numpy, pandas, matplotlib, yaml, wandb, tqdm, scipy, sklearn, dill, pydmd

For detailed help, see README_EXECUTION.txt
"""

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
matplotlib.use('Agg')  # Non-interactive backend for server execution
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


# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

class Config:
    """Configuration container for all experiment settings"""
    
    def __init__(self, args):
        self.args = args
        # 1. Setup dei percorsi
        self.setup_paths()
        # 2. Caricamento Configurazione (YAML) - FONDAMENTALE per creare self.parameters
        self.load_experiment_config()
        # 3. Setup Device
        self.setup_device()
        # 4. Set thin parameter (spatial downsampling factor)
        self.thin = 2  # 432x432 -> 216x216 grid
        
    def setup_paths(self):
        """Setup all directory paths"""
        # 1. Input Base (Dati)
        self.base_path = Path(self.args.base_path)

        # 2. Project Path (Codice)
        if hasattr(self.args, 'project_path') and self.args.project_path:
            self.project_path = Path(self.args.project_path)
        else:
            self.project_path = Path(".").resolve()

        # 3. Scratch Path (DMD input pesante)
        if hasattr(self.args, 'scratch_path') and self.args.scratch_path:
            self.scratch_path = Path(self.args.scratch_path)
        else:
            self.scratch_path = Path("/scratch_global/u10715220")

        # --- MODIFICA: GESTIONE OUTPUT SEPARATA ---
        # Se specifichiamo --output_dir, usiamo quello. Altrimenti usiamo base_path (Work).
        if self.args.output_dir:
            self.output_root = Path(self.args.output_dir)
            print(f"🚀 OUTPUTS REDIRECTED TO: {self.output_root}")
        else:
            self.output_root = self.base_path

        # Cartelle INPUT (restano su Work/Base)
        self.data_path = self.base_path / "data" / "ice"

        # Cartelle OUTPUT (vanno su Output Root -> Scratch)
        self.checkpoint_dir = self.output_root / "checkpoints"
        self.results_dir = self.output_root / "Results"
        self.scaler_dir = self.output_root / "scalers"
        # ------------------------------------------

        # Crea cartelle output se non esistono
        for dir_path in [self.checkpoint_dir, self.results_dir, self.scaler_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"📂 Directories configured:")
        print(f"   Inputs (Data):  {self.data_path}")
        print(f"   Outputs (Res):  {self.results_dir}")
        
    def load_experiment_config(self):
        """Load experiment configuration from YAML"""
        experiment_name = self.args.experiment_name
        experiment_number = self.args.experiment_number
        self.experiment = f"{experiment_name}_{experiment_number}"
        
        # Percorso del file YAML
        config_path = self.project_path / "multifidelity_transformer" / "experiment_configurations" / experiment_name / f"{experiment_number}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"❌ ERRORE CRITICO: Config non trovato in: {config_path}")

        print(f"📖 Loading config from: {config_path}")
        with open(config_path, 'r') as file:
            # QUESTA è la riga che mancava o falliva:
            self.parameters = yaml.safe_load(file)
            
        if self.parameters is None:
            raise ValueError(f"❌ Il file YAML è vuoto o non valido: {config_path}")
            
        print(f"✓ Config Loaded. Keys found: {list(self.parameters.keys())}")

    def setup_device(self):
        """Setup computation device"""
        if self.args.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.args.device
        print(f"🖥️  Using device: {self.device}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class DataLoader_Ice:
    """Handle loading and preprocessing of ice concentration data"""
    
    def __init__(self, config):
        self.config = config
        self.data_file = config.data_path / "Antarctic_years_1989_2024i.pkl"
        self.thickness_file = config.data_path / "Antarctic_thickness_1993_2023.pkl"
        self.sst_file = config.data_path / "Antarctic_SST_1993_2023.pkl"
        
    def load_raw_data(self):
        """Load raw ice concentration data"""
        print("\n" + "="*70)
        print("LOADING RAW DATA")
        print("="*70)
        
        with open(self.data_file, 'rb') as f:
            mask_land_, mask_ice_, data_, data_mean_month_, data_mean_week_, x_, y_ = dill.load(f)
            
        print(f"✓ Loaded {len(data_)} years of ice concentration data")
        print(f"  Original spatial dimensions: {data_[0].shape[1:]}") 
        
        # Load thickness and SST data
        with open(self.thickness_file, 'rb') as f:
            thickness_data_, thickness_years_ = dill.load(f)
        print(f"✓ Loaded {len(thickness_data_)} years of ice thickness data")
        
        with open(self.sst_file, 'rb') as f:
            sst_data_, sst_years_ = dill.load(f)
        print(f"✓ Loaded {len(sst_data_)} years of SST data")
        
        return mask_land_, mask_ice_, data_, data_mean_month_, data_mean_week_, x_, y_, thickness_data_, sst_data_
        
    def preprocess_data(self, mask_land_, mask_ice_, data_, data_mean_month_, data_mean_week_, x_, y_, thickness_data_, sst_data_):
        """Apply thinning and remove leap days"""
        print("\nPREPROCESSING DATA")
        print("-"*70)
        
        thin = self.config.parameters.get('data', {}).get('thin', 1)  # Default to 1 for full 432x432 grid
        
        data, data_mean_month, data_mean_week, x, y, mask_ice, mask_land = \
            thin_data(thin, data_, data_mean_month_, data_mean_week_, x_, y_, mask_ice_, mask_land_)
        
        data = del_leap(data)
        ny, nx = data[0].shape[1:]
        
        print(f"✓ Applied thinning factor: {thin}")
        print(f"  New spatial dimensions: ({ny}, {nx})")
        print(f"  Removed leap days from ice concentration")
        
        # Apply same preprocessing to thickness and SST data
        thickness_data = thin_data(thin, thickness_data_)[0]
        sst_data = thin_data(thin, sst_data_)[0]
        thickness_data = del_leap(thickness_data)
        sst_data = del_leap(sst_data)
        
        print(f"✓ Processed thickness data: {len(thickness_data)} years")
        print(f"✓ Processed SST data: {len(sst_data)} years")
        
        # Crop to bounding box around ice mask
        print("\n" + "-"*70)
        print("SPATIAL CROPPING TO ICE MASK BOUNDING BOX")
        print("-"*70)
        
        # Use ice mask directly (not active mask based on probability)
        # This includes all pixels in the static ice mask, not just high-probability regions
        
        # Compute bounding box with padding
        padding = 10  # pixels
        rows = np.any(mask_ice, axis=1)
        cols = np.any(mask_ice, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding and clip to grid bounds
        rmin = max(0, rmin - padding)
        rmax = min(ny - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(nx - 1, cmax + padding)
        
        ny_crop = rmax - rmin + 1
        nx_crop = cmax - cmin + 1
        
        print(f"  Original grid: ({ny}, {nx})")
        print(f"  Bounding box: rows [{rmin}:{rmax+1}], cols [{cmin}:{cmax+1}]")
        print(f"  Cropped grid: ({ny_crop}, {nx_crop})")
        print(f"  Reduction: {(1 - (ny_crop*nx_crop)/(ny*nx))*100:.1f}% fewer pixels")
        
        # Crop all data
        data = [year[:, rmin:rmax+1, cmin:cmax+1] for year in data]
        thickness_data = [year[:, rmin:rmax+1, cmin:cmax+1] for year in thickness_data]
        sst_data = [year[:, rmin:rmax+1, cmin:cmax+1] for year in sst_data]
        
        # Crop masks and coordinates
        mask_ice = mask_ice[rmin:rmax+1, cmin:cmax+1]
        mask_land = mask_land[rmin:rmax+1, cmin:cmax+1]
        x = x[cmin:cmax+1]
        y = y[rmin:rmax+1]
        
        # Update dimensions
        ny, nx = ny_crop, nx_crop
        
        print(f"✓ All data cropped to bounding box")
        
        # Store bounding box for DMD cropping later
        bbox = {'rmin': rmin, 'rmax': rmax, 'cmin': cmin, 'cmax': cmax}
        
        return data, data_mean_month, data_mean_week, x, y, mask_ice, mask_land, ny, nx, thickness_data, sst_data, bbox
        
    def split_train_val_test(self, data):
        """Robust train/val/test split by whole years"""
        print("\nTRAIN/VAL/TEST SPLIT")
        print("-"*70)
        
        # Inspect raw year shapes
        year_shapes = [tuple(np.asarray(y).shape) for y in data]
        shape_counts = Counter(year_shapes)
        
        if not shape_counts:
            raise ValueError("No yearly arrays found in data")
            
        target_shape, target_count = shape_counts.most_common(1)[0]
        print(f"Dominant year shape: {target_shape} × {target_count} years")
        
        # Filter to consistent shape
        filtered_years = [np.asarray(y) for y in data if tuple(np.asarray(y).shape) == target_shape]
        
        if len(target_shape) != 3:
            raise ValueError(f"Expected (days, nx, ny), got {target_shape}")
        if target_shape[0] != 365:
            raise ValueError(f"Expected 365 days per year, got {target_shape[0]}")
            
        years_available = len(filtered_years)
        
        # Split configuration
        DESIRED_TRAIN = 23
        DESIRED_VAL = 4
        DESIRED_TEST = 4
        
        if DESIRED_TRAIN > years_available:
            raise ValueError(f"Requested {DESIRED_TRAIN} training years but only {years_available} available")
            
        n_year_train = DESIRED_TRAIN
        n_year_val = DESIRED_VAL
        n_year_test = DESIRED_TEST
        
        # Stack and split
        data_years = np.stack(filtered_years, axis=0)
        
        data_train = data_years[:n_year_train]
        data_val = data_years[n_year_train:n_year_train + n_year_val]
        data_test = data_years[n_year_train + n_year_val:n_year_train + n_year_val + n_year_test]
        
        ny, nx = data_train.shape[2], data_train.shape[3]
        data_tot = np.concatenate((data_train, data_val, data_test), axis=0)
        
        print(f"✓ Split complete:")
        print(f"  Train: {data_train.shape}")
        print(f"  Val:   {data_val.shape}")
        print(f"  Test:  {data_test.shape}")
        print(f"  Total: {data_tot.shape}")
        
        return data_train, data_val, data_test, data_tot, n_year_train, n_year_val, n_year_test, ny, nx


class ClimatologyBaseline:
    """Compute climatology baselines for comparison"""
    
    def __init__(self, data_tot, x, y):
        self.data_tot = data_tot
        self.x = x
        self.y = y
        
    def compute_climatology(self):
        """Compute simple climatology (average over all years)"""
        print("\nCOMPUTING CLIMATOLOGY BASELINE")
        print("-"*70)
        
        climatology = self.data_tot.mean(axis=0)
        climatology_integral = np.trapz(np.trapz(climatology, self.y, axis=1), self.x, axis=1)
        
        print(f"✓ Climatology computed: {climatology.shape}")
        
        return climatology, climatology_integral
        
    def compute_progressive_climatology(self, years_memory=5):
        """Compute progressive climatology with rolling window"""
        print(f"\nCOMPUTING PROGRESSIVE CLIMATOLOGY (Memory: {years_memory} years)")
        print("-"*70)
        
        climatology_progressive = []
        n_years = len(self.data_tot)
        
        for i in range(n_years):
            start = 0 if i < years_memory else (i + 1 - years_memory)
            end = i + 1
            climatology_progressive.append(self.data_tot[start:end].mean(axis=0))
            
        climatology_progressive = np.array(climatology_progressive)
        climatology_progressive_integral = np.trapz(
            np.trapz(climatology_progressive, self.y, axis=2), self.x, axis=2
        )
        
        print(f"✓ Progressive climatology computed: {climatology_progressive.shape}")
        
        return climatology_progressive, climatology_progressive_integral


class DMDBaseline:
    """Load and prepare DMD baseline predictions"""
    
    def __init__(self, config, bbox=None):
        self.config = config
        self.bbox = bbox
        self.dmd_file = Path("/scratch_global/u10715220/checkpoints/dmd_forecasts_rank5_bootstrap100_years2-34.pkl")
        
    def load_dmd_predictions(self):
        """Load pre-computed DMD predictions and crop to bounding box"""
        print("\nLOADING DMD BASELINE")
        print("-"*70)
        
        if not self.dmd_file.exists():
            print(f"⚠️  DMD file not found: {self.dmd_file}")
            print("   Skipping DMD baseline (will use climatology only)")
            return None, None, None
            
        with open(self.dmd_file, 'rb') as f:
            dmd_results = dill.load(f)
            
        years = dmd_results['years']
        y_pred_mean = dmd_results['y_pred_mean']
        y_pred_std = dmd_results['y_pred_std']
        
        # Clamp predictions to [0, 1]
        y_pred_mean = np.clip(y_pred_mean, 0, 1)
        
        print(f"✓ DMD predictions loaded: {y_pred_mean.shape}")
        
        # Crop to same bounding box as data
        if self.bbox is not None:
            rmin, rmax = self.bbox['rmin'], self.bbox['rmax']
            cmin, cmax = self.bbox['cmin'], self.bbox['cmax']
            
            print(f"  Cropping DMD to bounding box: rows [{rmin}:{rmax+1}], cols [{cmin}:{cmax+1}]")
            y_pred_mean = y_pred_mean[:, :, rmin:rmax+1, cmin:cmax+1]
            if y_pred_std is not None:
                y_pred_std = y_pred_std[:, :, rmin:rmax+1, cmin:cmax+1]
            
            print(f"  Cropped DMD shape: {y_pred_mean.shape}")
        
        print(f"  Years covered: {years}")
        
        return years, y_pred_mean, y_pred_std


class LowFidelityDataPrep:
    """Prepare low-fidelity input data (ice thickness, SST, sensors)"""
    
    def __init__(self, config, data_tot, thickness_data, sst_data, x, y, mask_ice, mask_land, region_mask):
        self.config = config
        self.data_tot = data_tot
        self.thickness_data = thickness_data
        self.sst_data = sst_data
        self.x = x
        self.y = y
        self.mask_ice = mask_ice
        self.mask_land = mask_land
        self.region_mask = region_mask
        
    def prepare_thickness_data(self, n_year_train, n_year_val, n_year_test, n_year_tot):
        """Prepare ice thickness as level 1 input - FULL GRID (no masking)"""
        print("\nPREPARING LEVEL 1: ICE THICKNESS (FULL GRID)")
        print("-"*70)
        
        # Get total years and split
        thickness_tot = np.array(self.thickness_data[:n_year_tot])
        thickness_train = thickness_tot[:n_year_train]
        thickness_val = thickness_tot[n_year_train:n_year_train + n_year_val]
        thickness_test = thickness_tot[n_year_train + n_year_val:n_year_train + n_year_val + n_year_test]
        
        # Flatten FULL spatial grid (no region mask - keep all ny*nx pixels)
        # Shape: (n_years * 365, ny * nx)
        x1_train = thickness_train.reshape(-1, thickness_train.shape[2] * thickness_train.shape[3])
        x1_val = thickness_val.reshape(-1, thickness_val.shape[2] * thickness_val.shape[3])
        x1_test = thickness_test.reshape(-1, thickness_test.shape[2] * thickness_test.shape[3])
        
        ny, nx = thickness_train.shape[2], thickness_train.shape[3]
        print(f"✓ Level 1 (Thickness) shapes: Train{x1_train.shape}, Val{x1_val.shape}, Test{x1_test.shape}")
        print(f"  Grid dimensions: {ny} x {nx} = {ny*nx} pixels")
        
        return x1_train, x1_val, x1_test
        
    def prepare_sst_data(self, n_year_train, n_year_val, n_year_test, n_year_tot):
        """Prepare SST as level 2 input - FULL GRID (no masking)"""
        print("\nPREPARING LEVEL 2: SEA SURFACE TEMPERATURE (FULL GRID)")
        print("-"*70)
        
        # Get total years and split
        sst_tot = np.array(self.sst_data[:n_year_tot])
        sst_train = sst_tot[:n_year_train]
        sst_val = sst_tot[n_year_train:n_year_train + n_year_val]
        sst_test = sst_tot[n_year_train + n_year_val:n_year_train + n_year_val + n_year_test]
        
        # Flatten FULL spatial grid (no region mask - keep all ny*nx pixels)
        # Shape: (n_years * 365, ny * nx)
        x2_train = sst_train.reshape(-1, sst_train.shape[2] * sst_train.shape[3])
        x2_val = sst_val.reshape(-1, sst_val.shape[2] * sst_val.shape[3])
        x2_test = sst_test.reshape(-1, sst_test.shape[2] * sst_test.shape[3])
        
        ny, nx = sst_train.shape[2], sst_train.shape[3]
        print(f"✓ Level 2 (SST) shapes: Train{x2_train.shape}, Val{x2_val.shape}, Test{x2_test.shape}")
        print(f"  Grid dimensions: {ny} x {nx} = {ny*nx} pixels")
        
        return x2_train, x2_val, x2_test
    
    def apply_pod_reduction(self, x1_train, x1_val, x1_test,
                            x2_train, x2_val, x2_test,
                            threshold=0.95):
        """Apply POD (Proper Orthogonal Decomposition) to reduce x1 and x2 dimensions
        
        POD is computed on the FULL spatial grid for maximum information retention,
        then reduced based on variance threshold or fixed number of modes.
        
        Args:
            x1_train, x1_val, x1_test: Ice thickness data (time_steps, ny*nx)
            x2_train, x2_val, x2_test: SST data (time_steps, ny*nx)
            threshold: Either int (n_POD directly) or float (variance threshold, default: 0.95)
            
        Returns:
            Reduced data for all splits, POD basis and singular values
        """
        print("\nAPPLYING POD DECOMPOSITION")
        print("="*70)
        
        # Get dimensions
        N_dof_x1 = x1_train.shape[1]  # Number of spatial pixels for thickness
        N_dof_x2 = x2_train.shape[1]  # Number of spatial pixels for SST
        n_years_train = x1_train.shape[0]
        n_years_val = x1_val.shape[0]
        n_years_test = x1_test.shape[0]
        
        n_POD_large = 128  # Initial large number for SVD computation
        
        print(f"\nOriginal dimensions:")
        print(f"  Ice Thickness: {N_dof_x1} pixels")
        print(f"  SST: {N_dof_x2} pixels")
        print(f"  Initial SVD modes: {n_POD_large}")
        
        # ====================================================================
        # POD for Level 1 (Ice Thickness)
        # ====================================================================
        print("\n" + "-"*70)
        print("LEVEL 1: ICE THICKNESS POD")
        print("-"*70)
        
        # Reshape for POD: transpose to (n_pixels, n_timesteps)
        x1_train_pod = np.reshape(x1_train.T, (N_dof_x1, -1), 'F')
        x1_val_pod = np.reshape(x1_val.T, (N_dof_x1, -1), 'F')
        x1_test_pod = np.reshape(x1_test.T, (N_dof_x1, -1), 'F')
        
        print(f"POD input shapes: train{x1_train_pod.shape}, val{x1_val_pod.shape}, test{x1_test_pod.shape}")
        
        # Compute POD on training data ONLY (no data leakage)
        print("Computing randomized SVD on training data...")
        U1, S1 = compute_randomized_SVD(x1_train_pod, n_POD_large, N_dof_x1, 1)
        
        # Determine n_POD based on threshold
        if isinstance(threshold, int):
            n_POD_x1 = threshold
            print(f"Number of POD modes: {n_POD_x1} (fixed)")
        elif isinstance(threshold, float):
            var_threshold = threshold
            n_POD_x1 = np.argmax(np.cumsum(S1)/np.sum(S1) > var_threshold) + 1
            print(f"Number of POD modes: {n_POD_x1} (variance threshold: {var_threshold})")
        else:
            raise ValueError("Threshold must be an integer (n_POD) or a float (var_threshold).")
        
        # Trim to selected modes
        U1 = U1[:, :n_POD_x1]
        S1 = S1[:n_POD_x1]
        
        # Explained variance
        energy_captured = np.cumsum(S1) / np.sum(S1)
        print(f"✓ SVD complete:")
        print(f"  Energy captured by {n_POD_x1} modes: {energy_captured[-1]*100:.2f}%")
        print(f"  Singular values range: [{S1[0]:.2e}, {S1[-1]:.2e}]")
        
        # Project ALL data onto POD basis
        print("Projecting data onto POD basis...")
        x1_train_reduced = np.dot(x1_train, U1)  # (n_timesteps, n_POD)
        x1_val_reduced = np.dot(x1_val, U1)
        x1_test_reduced = np.dot(x1_test, U1)
        
        # REMOVED: Singular value normalization (will use RobustScaler instead)
        # print("Scaling POD coefficients by singular values...")
        # x1_train_reduced = x1_train_reduced / S1[:n_POD_x1]
        # x1_val_reduced = x1_val_reduced / S1[:n_POD_x1]
        # x1_test_reduced = x1_test_reduced / S1[:n_POD_x1]
        
        print(f"✓ Reduced shapes: train{x1_train_reduced.shape}, val{x1_val_reduced.shape}, test{x1_test_reduced.shape}")
        
        # ====================================================================
        # POD for Level 2 (SST)
        # ====================================================================
        print("\n" + "-"*70)
        print("LEVEL 2: SEA SURFACE TEMPERATURE POD")
        print("-"*70)
        
        # Reshape for POD: transpose to (n_pixels, n_timesteps)
        x2_train_pod = np.reshape(x2_train.T, (N_dof_x2, -1), 'F')
        x2_val_pod = np.reshape(x2_val.T, (N_dof_x2, -1), 'F')
        x2_test_pod = np.reshape(x2_test.T, (N_dof_x2, -1), 'F')
        
        print(f"POD input shapes: train{x2_train_pod.shape}, val{x2_val_pod.shape}, test{x2_test_pod.shape}")
        
        # Compute POD on training data ONLY (no data leakage)
        print("Computing randomized SVD on training data...")
        U2, S2 = compute_randomized_SVD(x2_train_pod, n_POD_large, N_dof_x2, 1)
        
        # Determine n_POD based on threshold
        if isinstance(threshold, int):
            n_POD_x2 = threshold
            print(f"Number of POD modes: {n_POD_x2} (fixed)")
        elif isinstance(threshold, float):
            var_threshold = threshold
            n_POD_x2 = np.argmax(np.cumsum(S2)/np.sum(S2) > var_threshold) + 1
            print(f"Number of POD modes: {n_POD_x2} (variance threshold: {var_threshold})")
        else:
            raise ValueError("Threshold must be an integer (n_POD) or a float (var_threshold).")
        
        # Trim to selected modes
        U2 = U2[:, :n_POD_x2]
        S2 = S2[:n_POD_x2]
        
        # Explained variance
        energy_captured = np.cumsum(S2) / np.sum(S2)
        print(f"✓ SVD complete:")
        print(f"  Energy captured by {n_POD_x2} modes: {energy_captured[-1]*100:.2f}%")
        print(f"  Singular values range: [{S2[0]:.2e}, {S2[-1]:.2e}]")
        
        # Project ALL data onto POD basis
        print("Projecting data onto POD basis...")
        x2_train_reduced = np.dot(x2_train, U2)  # (n_timesteps, n_POD)
        x2_val_reduced = np.dot(x2_val, U2)
        x2_test_reduced = np.dot(x2_test, U2)
        
        # REMOVED: Singular value normalization (will use RobustScaler instead)
        # print("Scaling POD coefficients by singular values...")
        # x2_train_reduced = x2_train_reduced / S2[:n_POD_x2]
        # x2_val_reduced = x2_val_reduced / S2[:n_POD_x2]
        # x2_test_reduced = x2_test_reduced / S2[:n_POD_x2]
        
        print(f"✓ Reduced shapes: train{x2_train_reduced.shape}, val{x2_val_reduced.shape}, test{x2_test_reduced.shape}")
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "="*70)
        print("POD REDUCTION SUMMARY")
        print("="*70)
        print(f"Dimension reduction:")
        print(f"  Level 1 (Thickness): {N_dof_x1} → {n_POD_x1} ({n_POD_x1/N_dof_x1*100:.1f}% of original)")
        print(f"  Level 2 (SST):       {N_dof_x2} → {n_POD_x2} ({n_POD_x2/N_dof_x2*100:.1f}% of original)")
        print(f"  Speedup factor:      ~{N_dof_x1/n_POD_x1:.1f}x")
        print("="*70)
        
        # Store POD metadata
        pod_data = {
            'n_POD_x1': n_POD_x1,
            'n_POD_x2': n_POD_x2,
            'U1': U1, 'S1': S1,
            'U2': U2, 'S2': S2,
            'N_dof_x1': N_dof_x1,
            'N_dof_x2': N_dof_x2
        }
        
        return (x1_train_reduced, x1_val_reduced, x1_test_reduced,
                x2_train_reduced, x2_val_reduced, x2_test_reduced,
                pod_data)
        
    def define_target_region(self, x_min=-0.15, x_max=-0.05, y_min=0.05, y_max=0.15):
        """Define the target region mask (for consistency with old code)"""
        print("\nDEFINING TARGET REGION")
        print("-"*70)
        
        x_region = (self.x >= x_min) & (self.x <= x_max)
        y_region = (self.y >= y_min) & (self.y <= y_max)
        
        ny, nx = self.mask_land.shape
        region_mask_x = np.zeros((ny, nx), dtype=bool)
        region_mask_y = np.zeros((ny, nx), dtype=bool)
        region_mask_x[:, x_region] = True
        region_mask_y[y_region, :] = True
        region_mask = region_mask_x & region_mask_y & (~self.mask_land)
        
        n_pixel_region = region_mask.sum()
        
        print(f"✓ Region defined: [{x_min}, {x_max}] × [{y_min}, {y_max}]")
        print(f"  Pixels in region: {n_pixel_region}")
        
        return region_mask, n_pixel_region
        
    def prepare_sensors(self, y_true_data, y_dmd_pred, dmd_years, n_sensors=128, seed=0, sensor_noise_std=0.05):
        """Sample DMD residuals at random sensor locations in the active mask
        
        Instead of using ground truth measurements, we sample DMD residuals (Y_TRUE - Y_DMD)
        at randomly placed sensor locations within the active region mask.
        
        Args:
            y_true_data: Full true ice concentration data (all years)
            y_dmd_pred: DMD predictions matching dmd_years
            dmd_years: Years covered by DMD predictions
            n_sensors: Number of sensor locations to sample (default: 128)
            seed: Random seed for reproducible sensor placement
            sensor_noise_std: Standard deviation of Gaussian noise to add to sensors (default: 0.05)
            
        Returns:
            sensor_residuals: (n_years, n_days, n_sensors) - DMD residuals at sensor locations with noise
            sensor_mask: Boolean mask showing sensor locations
            sensor_idxs: Array indices of sensor locations
            n_sensors: Actual number of sensors (may be less if region is small)
        """
        print(f"\nPREPARING LEVEL 3: SENSORS (n={n_sensors}, sampling DMD residuals with noise)")
        print("-"*70)
        print(f"  Sensor noise (std): {sensor_noise_std}")
        
        # Align true data with DMD years
        if len(y_true_data) == len(dmd_years):
            print(f"  Data length ({len(y_true_data)}) matches DMD years. Using direct alignment.")
            y_true_aligned = np.array(y_true_data)
        else:
            print(f"  Data length ({len(y_true_data)}) != DMD years ({len(dmd_years)}). Using indexing.")
            y_true_aligned = np.array([y_true_data[year] for year in dmd_years])
        
        # Compute residuals: Y_TRUE - Y_DMD (full spatial grid)
        residuals_full = y_true_aligned - y_dmd_pred
        
        # Clip residuals to [-1, 1] (physical constraint: ice concentration is in [0, 1])
        residuals_full = np.clip(residuals_full, -1, 1)
        
        # Get available sensor locations from region mask
        mask_region_idxs = np.argwhere(self.region_mask)
        
        print(f"  Available sensor locations in active mask: {len(mask_region_idxs)}")
        
        # Random sensor placement within active region
        np.random.seed(seed)
        if len(mask_region_idxs) >= n_sensors:
            sensor_idxs = mask_region_idxs[
                np.random.choice(mask_region_idxs.shape[0], n_sensors, replace=False)
            ]
        else:
            print(f"  ⚠️  Only {len(mask_region_idxs)} valid locations, using all")
            sensor_idxs = mask_region_idxs
            n_sensors = len(sensor_idxs)
        
        # Create sensor mask
        sensor_mask = np.zeros_like(self.region_mask, dtype=bool)
        sensor_mask[tuple(sensor_idxs.T)] = True
        
        # Extract residuals at sensor locations
        sensor_residuals = residuals_full[:, :, sensor_mask]
        
        # Add Gaussian noise to simulate sensor measurement uncertainty
        np.random.seed(seed)  # Use same seed for reproducibility
        noise = np.random.normal(0, sensor_noise_std, sensor_residuals.shape)
        sensor_residuals_noisy = sensor_residuals + noise
        
        # Clip noisy residuals to maintain physical constraints
        sensor_residuals_noisy = np.clip(sensor_residuals_noisy, -1, 1)
        
        print(f"✓ Sensors placed: {n_sensors}")
        print(f"  Data shape: {sensor_residuals_noisy.shape} (years × days × sensors)")
        print(f"  Residual statistics at sensors (with noise):")
        print(f"    Mean: {sensor_residuals_noisy.mean():.6f}")
        print(f"    Std:  {sensor_residuals_noisy.std():.6f}")
        print(f"    Min:  {sensor_residuals_noisy.min():.6f}")
        print(f"    Max:  {sensor_residuals_noisy.max():.6f}")
        print(f"  Original (no noise) - Mean: {sensor_residuals.mean():.6f}, Std: {sensor_residuals.std():.6f}")
        
        return sensor_residuals_noisy, sensor_mask, sensor_idxs, n_sensors
        
    def split_sensor_data(self, sensor_residuals, n_year_train, n_year_val, n_year_test):
        """Split sensor data (DMD residuals at sensor locations) into train/val/test"""
        print("\nSPLITTING SENSOR DATA (DMD RESIDUALS)")
        print("-"*70)
        
        x3_train = sensor_residuals[:n_year_train]
        x3_val = sensor_residuals[n_year_train:n_year_train + n_year_val]
        x3_test = sensor_residuals[n_year_train + n_year_val:n_year_train + n_year_val + n_year_test]
        
        # Flatten time dimension
        x3_train = x3_train.reshape(-1, x3_train.shape[-1])
        x3_val = x3_val.reshape(-1, x3_val.shape[-1])
        x3_test = x3_test.reshape(-1, x3_test.shape[-1])
        
        print(f"✓ Level 3 (Sensors - DMD Residuals): Train{x3_train.shape}, Val{x3_val.shape}, Test{x3_test.shape}")
        print(f"  Train range: [{x3_train.min():.4f}, {x3_train.max():.4f}]")
        print(f"  Val range:   [{x3_val.min():.4f}, {x3_val.max():.4f}]")
        print(f"  Test range:  [{x3_test.min():.4f}, {x3_test.max():.4f}]")
        
        return x3_train, x3_val, x3_test 


# ============================================================================
# PART 2 COMPLETE
# ============================================================================


class HighFidelityTargetPrep:
    """Prepare high-fidelity target data (DMD residuals)"""
    
    def __init__(self, config, data, y_dmd_pred, dmd_years, region_mask, n_year_train, n_year_val, n_year_test):
        self.config = config
        self.data = data
        self.y_dmd_pred = y_dmd_pred
        self.dmd_years = dmd_years
        self.region_mask = region_mask
        self.n_year_train = n_year_train
        self.n_year_val = n_year_val
        self.n_year_test = n_year_test
        
    def compute_residuals(self):
        """Compute residuals between DMD predictions and true data"""
        print("\nCOMPUTING HIGH-FIDELITY RESIDUALS")
        print("-"*70)
        
        # Get true data for DMD years
        # --- FIX INDICI ---
        if len(self.data) == len(self.dmd_years):
            print(f"DEBUG: Data length ({len(self.data)}) matches DMD years. Using direct alignment.")
            y_true = np.array(self.data)
        else:
            print(f"DEBUG: Data length ({len(self.data)}) != DMD years ({len(self.dmd_years)}). Using indexing.")
            y_true = np.array([self.data[year] for year in self.dmd_years])
         # ------------------
            
            # Compute residuals
        y_dmd_residuals = y_true - self.y_dmd_pred
        
        print(f"✓ Residuals computed: {y_dmd_residuals.shape}")
        print(f"  Statistics:")
        print(f"    Mean: {y_dmd_residuals.mean():.6f}")
        print(f"    Std:  {y_dmd_residuals.std():.6f}")
        print(f"    Min:  {y_dmd_residuals.min():.6f}")
        print(f"    Max:  {y_dmd_residuals.max():.6f}")
        
        return y_dmd_residuals, y_true
        
    def split_and_normalize_residuals(self, y_dmd_residuals):
        """Split residuals and normalize using ONLY ice mask statistics
        
        CRITICAL: Statistics (mean/std) are computed ONLY on ice mask pixels
        to ensure consistency with:
        - Training loss (computed on ice mask)
        - Conformal calibration (computed on ice mask)
        - Final predictions (evaluated on ice mask)
        """
        print("\nSPLITTING AND NORMALIZING RESIDUALS (ICE MASK ONLY)")
        print("-"*70)
        
        # Split by time
        residuals_train = y_dmd_residuals[:self.n_year_train]
        residuals_val = y_dmd_residuals[self.n_year_train:self.n_year_train + self.n_year_val]
        residuals_test = y_dmd_residuals[self.n_year_train + self.n_year_val:self.n_year_train + self.n_year_val + self.n_year_test]
        
        print(f"✓ Split residuals (full spatial):")
        print(f"  Train: {residuals_train.shape}")
        print(f"  Val:   {residuals_val.shape}")
        print(f"  Test:  {residuals_test.shape}")
        
        # Extract ONLY ice mask pixels before computing statistics
        print(f"\n✓ Extracting ice mask pixels for normalization...")
        residuals_train_masked = residuals_train[:, :, self.region_mask]
        residuals_val_masked = residuals_val[:, :, self.region_mask]
        residuals_test_masked = residuals_test[:, :, self.region_mask]
        
        n_pixels_mask = residuals_train_masked.shape[2]
        print(f"  Pixels in ice mask: {n_pixels_mask}")
        print(f"  Train (masked): {residuals_train_masked.shape}")
        print(f"  Val (masked):   {residuals_val_masked.shape}")
        print(f"  Test (masked):  {residuals_test_masked.shape}")
        
        # Compute statistics ONLY on ice mask pixels from training set
        train_mean = residuals_train_masked.mean()
        train_std = residuals_train_masked.std()
        
        print(f"\n✓ Normalization statistics (from ice mask pixels only):")
        print(f"  Train mean: {train_mean:.6f}")
        print(f"  Train std:  {train_std:.6f}")
        
        # Normalize masked data using ice-mask-only statistics
        y_train = (residuals_train_masked - train_mean) / train_std
        y_val = (residuals_val_masked - train_mean) / train_std
        y_test = (residuals_test_masked - train_mean) / train_std
        
        print(f"\n✓ Normalized residuals (ice mask only):")
        print(f"  Train mean/std: {y_train.mean():.6f} / {y_train.std():.6f}")
        print(f"  Val mean/std:   {y_val.mean():.6f} / {y_val.std():.6f}")
        print(f"  Test mean/std:  {y_test.mean():.6f} / {y_test.std():.6f}")
        
        return y_train, y_val, y_test, train_mean, train_std
        
    def extract_region(self, y_train, y_val, y_test):
        """Verify region extraction (already done in split_and_normalize_residuals)
        
        NOTE: This method is now a verification step. The actual extraction of ice mask
        pixels is performed in split_and_normalize_residuals() to ensure statistics
        are computed only on the relevant pixels.
        """
        print("\nVERIFYING REGION EXTRACTION")
        print("-"*70)
        print("  ℹ️  Ice mask extraction already performed during normalization")
        print("     (ensures statistics are computed only on ice mask pixels)")
        
        # Data is already masked, just pass through and compute dimensions
        y_train_region = y_train
        y_val_region = y_val
        y_test_region = y_test
        
        n_pixel_region = y_train_region.shape[2]
        
        print(f"\n✓ Region data verified:")
        print(f"  Train: {y_train_region.shape}")
        print(f"  Val:   {y_val_region.shape}")
        print(f"  Test:  {y_test_region.shape}")
        print(f"  Pixels in region: {n_pixel_region}")
        
        return y_train_region, y_val_region, y_test_region, n_pixel_region


class DataScaler:
    """Robust scaling for all fidelity levels"""
    
    def __init__(self, config):
        self.config = config
        
    @staticmethod
    def get_flat_shape(arr):
        """Flatten array to 2D while preserving feature dimension"""
        if arr.ndim == 1:
            return arr.reshape(-1, 1), arr.shape
        if arr.ndim == 2:
            return arr, arr.shape
        # (Y, D, ...) -> (Y*D, ...)
        return arr.reshape(-1, np.prod(arr.shape[2:])), arr.shape
        
    def robust_scale_fit_transform(self, train_arr, val_arr, test_arr, chunk_size=4096):
        """Apply robust scaling (median/IQR) with memory-efficient chunking"""
        print("  Applying robust scaling...")
        
        # Flatten
        train_2d, train_shape = self.get_flat_shape(train_arr)
        val_2d, val_shape = self.get_flat_shape(val_arr)
        test_2d, test_shape = self.get_flat_shape(test_arr)
        
        n_feats = train_2d.shape[1]
        med = np.zeros(n_feats, dtype=np.float32)
        iqr = np.zeros(n_feats, dtype=np.float32)
        
        # Chunked processing
        for i in range(0, n_feats, chunk_size):
            end = min(n_feats, i + chunk_size)
            
            # Fit on train
            block = train_2d[:, i:end]
            m = np.median(block, axis=0)
            q1 = np.percentile(block, 25, axis=0)
            q3 = np.percentile(block, 75, axis=0)
            iq = q3 - q1
            iq[iq < 1e-6] = 1.0  # Avoid division by zero
            
            med[i:end] = m
            iqr[i:end] = iq
            
            # Transform all
            train_2d[:, i:end] = (train_2d[:, i:end] - m) / iq
            val_2d[:, i:end] = (val_2d[:, i:end] - m) / iq
            test_2d[:, i:end] = (test_2d[:, i:end] - m) / iq
            
        # Restore shapes
        return (
            train_2d.reshape(train_shape).astype(np.float32),
            val_2d.reshape(val_shape).astype(np.float32),
            test_2d.reshape(test_shape).astype(np.float32),
            med, iqr
        )
        
    def scale_all_levels(self, x1_train, x1_val, x1_test,
                        x2_train, x2_val, x2_test,
                        x3_train, x3_val, x3_test):
        """Scale all three fidelity levels
        
        All levels now use RobustScaler for consistent normalization and better
        handling of outliers. POD coefficients are no longer normalized by singular values.
        """
        print("\nSCALING FIDELITY LEVELS")
        print("-"*70)
        
        print("Level 1 (Ice Thickness POD): Applying robust scaling...")
        x1_train, x1_val, x1_test, med1, iqr1 = self.robust_scale_fit_transform(
            x1_train, x1_val, x1_test
        )
        
        print("Level 2 (SST POD): Applying robust scaling...")
        x2_train, x2_val, x2_test, med2, iqr2 = self.robust_scale_fit_transform(
            x2_train, x2_val, x2_test
        )
        
        print("Level 3 (Sensors): Applying robust scaling...")
        x3_train, x3_val, x3_test, med3, iqr3 = self.robust_scale_fit_transform(
            x3_train, x3_val, x3_test
        )
        
        # Store scalers
        scalers = {
            'level_1': {'median': med1, 'iqr': iqr1, 'method': 'robust_scaler'},
            'level_2': {'median': med2, 'iqr': iqr2, 'method': 'robust_scaler'},
            'level_3': {'median': med3, 'iqr': iqr3, 'method': 'robust_scaler'}
        }
        
        # Save scalers
        scaler_path = self.config.scaler_dir / 'scalers_multifidelity.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
            
        print(f"\n✓ Scaling complete. All levels use RobustScaler.")
        print(f"  Scalers saved to: {scaler_path}")
        
        return x1_train, x1_val, x1_test, x2_train, x2_val, x2_test, x3_train, x3_val, x3_test, scalers


# ============================================================================
# PART 3 COMPLETE
# ============================================================================


class TorchDatasetPreparation:
    """Prepare PyTorch datasets and dataloaders"""
    
    def __init__(self, config):
        self.config = config
        self.seq_len = config.parameters["data"]["seq_len"]
        self.seq_freq = config.parameters["data"]["seq_freq"]
        
    def prepare_tensor_train(self, data_array):
        """For TRAIN: Use unfold to create batched sequences"""
        t = torch.tensor(data_array, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        elif t.ndim == 3:
            t = t.reshape(-1, t.shape[-1])
        elif t.ndim == 4:
            t = t.reshape(-1, np.prod(t.shape[2:]))
            
        return t.unfold(0, self.seq_len, self.seq_freq).permute(0, 2, 1)
        
    def prepare_tensor_eval(self, data_array):
        """For VAL/TEST: Keep full continuous sequence"""
        t = torch.tensor(data_array, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        elif t.ndim == 3:
            t = t.reshape(-1, t.shape[-1])
        elif t.ndim == 4:
            t = t.reshape(-1, np.prod(t.shape[2:]))
            
        return t.unsqueeze(0)
        
    def create_sequences(self, x0_train, x1_train, x2_train, x3_train, y_train,
                        x0_val, x1_val, x2_val, x3_val, y_val,
                        x0_test, x1_test, x2_test, x3_test, y_test):
        """Create all sequence tensors"""
        print("\nCREATING SEQUENCE TENSORS")
        print("-"*70)
        
        # Training (batched sequences)
        print("Training sequences (batched)...")
        x0_train_seq = self.prepare_tensor_train(x0_train)
        x1_train_seq = self.prepare_tensor_train(x1_train)
        x2_train_seq = self.prepare_tensor_train(x2_train)
        x3_train_seq = self.prepare_tensor_train(x3_train)
        y_train_seq = self.prepare_tensor_train(y_train)
        
        # Validation (continuous)
        print("Validation sequences (continuous)...")
        x0_val_seq = self.prepare_tensor_eval(x0_val)
        x1_val_seq = self.prepare_tensor_eval(x1_val)
        x2_val_seq = self.prepare_tensor_eval(x2_val)
        x3_val_seq = self.prepare_tensor_eval(x3_val)
        y_val_seq = self.prepare_tensor_eval(y_val)
        
        # Test (continuous)
        print("Test sequences (continuous)...")
        x0_test_seq = self.prepare_tensor_eval(x0_test)
        x1_test_seq = self.prepare_tensor_eval(x1_test)
        x2_test_seq = self.prepare_tensor_eval(x2_test)
        x3_test_seq = self.prepare_tensor_eval(x3_test)
        y_test_seq = self.prepare_tensor_eval(y_test)
        
        print(f"\n✓ Sequences created:")
        print(f"  Train (batched): {x2_train_seq.shape}")
        print(f"  Val (continuous): {x2_val_seq.shape}")
        print(f"  Test (continuous): {x2_test_seq.shape}")
        
        return {
            'train': (x0_train_seq, x1_train_seq, x2_train_seq, x3_train_seq, y_train_seq),
            'val': (x0_val_seq, x1_val_seq, x2_val_seq, x3_val_seq, y_val_seq),
            'test': (x0_test_seq, x1_test_seq, x2_test_seq, x3_test_seq, y_test_seq)
        }
        
    def create_dataloaders(self, sequences):
        """Create PyTorch dataloaders"""
        print("\nCREATING DATALOADERS")
        print("-"*70)
        
        # Unpack sequences
        x0_train, x1_train, x2_train, x3_train, y_train = sequences['train']
        x0_val, x1_val, x2_val, x3_val, y_val = sequences['val']
        x0_test, x1_test, x2_test, x3_test, y_test = sequences['test']
        
        # Create feature dictionaries
        train_features = {
            "level_0": x0_train,
            "level_1": x1_train,
            "level_2": x2_train,
            "level_3": x3_train
        }
        
        val_features = {
            "level_0": x0_val,
            "level_1": x1_val,
            "level_2": x2_val,
            "level_3": x3_val
        }
        
        test_features = {
            "level_0": x0_test,
            "level_1": x1_test,
            "level_2": x2_test,
            "level_3": x3_test
        }
        
        # Create datasets
        # Training: Use all mask combinations (sequential_mask from config)
        train_dataset = MultiFidelityDataset(
            train_features, y_train, device='cpu',
            sequential=self.config.parameters["data"]["sequential_mask"]
        )
        
        # Validation: Use single default mask [F,F,F] since force_mask() overrides during calibration
        # This avoids 7× dataset expansion (only 1× size instead of 7×)
        val_dataset = MultiFidelityDataset(
            val_features, y_val, device='cpu',
            sequential=False,
            single_mask=(False, False, False)  # Default: all levels available
        )
        
        # Test: Use single default mask [F,F,F] since force_mask() overrides during evaluation
        # This avoids 7× dataset expansion (only 1× size instead of 7×)
        test_dataset = MultiFidelityDataset(
            test_features, y_test, device='cpu',
            sequential=False,
            single_mask=(False, False, False)  # Default: all levels available
        )
        
        # DATALOADERS (Safe Mode)
        # ---------------------------------------------------------
        print("🔧 Creating DataLoaders (num_workers=0, pin_memory=False)...")
        batch_size = self.config.parameters["training"]["batch_size"]
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        print("✓ DataLoaders ready.")
        
        print(f"✓ Dataloaders created (Batch size: {batch_size})")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        print(f"  Test batches:  {len(test_loader)}")
        
        return train_loader, val_loader, test_loader, train_dataset


class ModelSetup:
    """Setup model, loss function, and optimizer"""
    
    def __init__(self, config, levels_dim, n_pixel_region, train_dataset):
        self.config = config
        self.levels_dim = levels_dim
        self.n_pixel_region = n_pixel_region
        self.train_dataset = train_dataset
        self.quantiles = [0.05, 0.5, 0.95]
        
    def create_model(self):
        """Create multifidelity transformer model"""
        print("\nCREATING MODEL")
        print("-"*70)
        
        output_dim = self.n_pixel_region * len(self.quantiles)
        n_masks = (self.train_dataset.mask_manager.n_masks 
                  if hasattr(self.train_dataset, 'mask_manager') else 1)
        
        model = MultifidelityTransformer(
            levels_dim=self.levels_dim,
            embedding_dim=int(self.config.parameters["model"]["embedding_dim"]),
            parameters_dim=1,  # x0 has dimension 1
            output_dim=output_dim,
            n_heads=int(self.config.parameters["model"]["n_heads"]),
            n_masks=(n_masks if self.config.parameters["model"]["mask_embeddings"] else None),
            n_transformer_blocks=int(self.config.parameters["model"]["n_transformer_blocks"]),
            spatial_encoders_dim={},
            dropout=0.2  # Increased from 0.1 to 0.2 to prevent overfitting
        ).to(self.config.device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created with skip connections and 4×emb_dim FFN")
        print(f"  Parameters: {n_params:,}")
        print(f"  Output dimension: {output_dim}")
        print(f"  Quantiles: {self.quantiles}")
        print(f"  Architecture: {int(self.config.parameters['model']['n_transformer_blocks'])} Transformer blocks, emb_dim={int(self.config.parameters['model']['embedding_dim'])}, heads={int(self.config.parameters['model']['n_heads'])}")
        
        return model
        
    def create_loss_function(self):
        """Create full spatial quantile loss"""
        
        class FullSpatialQuantileLoss(nn.Module):
            def __init__(self, quantiles, n_pixels):
                super().__init__()
                self.quantiles = quantiles
                self.n_pixels = n_pixels
                
            def forward(self, preds, targets):
                B, S, _ = preds.shape
                preds_r = preds.view(B, S, len(self.quantiles), self.n_pixels)
                
                loss = 0.0
                for i, tau in enumerate(self.quantiles):
                    pred_q = preds_r[:, :, i, :]
                    err = targets - pred_q
                    # Compute quantile loss with in-place max to save memory
                    tau_weight = torch.where(err >= 0, tau, tau - 1)
                    loss += torch.mean(tau_weight * err)
                    # Free memory immediately
                    del pred_q, err, tau_weight
                    
                return loss
                
        loss_fn = FullSpatialQuantileLoss(self.quantiles, self.n_pixel_region)
        print(f"✓ Loss function created (Full region evaluation)")
        
        return loss_fn
        
    def create_optimizer_scheduler(self, model):
        """Create optimizer and learning rate scheduler"""
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.parameters["training"]["optim_params"]["learning_rate"]),
            weight_decay=float(self.config.parameters["training"]["optim_params"]["weight_decay"])
        )
        
        lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_schedule(
                step,
                int(self.config.parameters["model"]["embedding_dim"]),
                1.0,
                int(self.config.parameters["training"]["optim_params"]["warmup"])
            )
        )
        
        print(f"✓ Optimizer and scheduler created")
        
        return optimizer, lr_scheduler


# ============================================================================
# PART 4 COMPLETE
# ============================================================================


class ModelTrainer:
    """Handle model training with checkpointing and early stopping"""
    
    def __init__(self, config, model, loss_fn, optimizer, lr_scheduler, 
                 train_loader, val_loader):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Paths
        self.checkpoint_path = config.checkpoint_dir / f"{config.experiment}.pt"
        self.latest_path = config.checkpoint_dir / f"{config.experiment}_latest.pt"
        self.history_path = config.checkpoint_dir / f"{config.experiment}_history.pkl"
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.combinatorial_patterns = ["3L_all", "2L_1+2", "2L_1+3", "2L_2+3", "1L_1", "1L_2", "1L_3"]
        self.train_fidelity_losses = {p: [] for p in self.combinatorial_patterns}
        self.val_fidelity_losses = {p: [] for p in self.combinatorial_patterns}
        self.best_val_loss = float('inf')
        self.epochs_trained = 0
        
        # WandB
        self.wandb_run = None
        if config.parameters["logging"]["wandb"] and not config.args.no_wandb:
            wandb.login()
            self.wandb_run = wandb.init(
                project="MultifidelityTransformer",
                config=config.parameters,
                name=config.parameters["experiment_name"],
                reinit=True
            )
            
    def load_checkpoint(self):
        """Load model checkpoint if it exists"""
        if not self.config.parameters["model"]["load_model"]:
            return False
            
        print("\n" + "="*70)
        print("LOADING CHECKPOINT")
        print("="*70)
        
        if self.checkpoint_path.exists():
            state_dict = torch.load(self.checkpoint_path, map_location=self.config.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"✓ Model weights loaded from: {self.checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        if self.history_path.exists():
            with open(self.history_path, 'rb') as f:
                history = pickle.load(f)
                
            self.train_losses = history.get('train_loss', [])
            self.val_losses = history.get('val_loss', [])
            self.train_fidelity_losses = history.get('train_fidelity', {})
            self.val_fidelity_losses = history.get('val_fidelity', {})
            self.best_val_loss = history.get('best_val_loss', float('inf'))
            self.epochs_trained = history.get('epochs_trained', 0)
            
            print(f"✓ Training history loaded: {self.epochs_trained} epochs")
        else:
            print("⚠️  No training history found")
            
        return True
        
    def save_history(self):
        """Save training history"""
        history = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_fidelity": self.train_fidelity_losses,
            "val_fidelity": self.val_fidelity_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.epochs_trained,
            "parameters": self.config.parameters
        }
        
        with open(self.history_path, 'wb') as f:
            pickle.dump(history, f)
            
    def train(self):
        """Main training loop"""
        if self.load_checkpoint():
            print("\n✓ Model loaded. Skipping training.")
            return
            
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        epochs = int(self.config.parameters["training"]["epochs"])
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        print(f"Epochs: {epochs}, Patience: {patience}")
        print("Press Ctrl+C to interrupt and save progress")
        
        # Watch model with WandB
        if self.wandb_run and self.config.parameters["logging"]["gradients"]:
            self.wandb_run.watch(self.model, log="all", log_freq=100)
            
        try:
            for epoch in range(epochs):
                # Train
                self.model.train()
                tr_res = run_epoch(
                    self.train_loader, self.model, self.loss_fn,
                    self.optimizer, self.lr_scheduler, 10,
                    epoch_n=epoch, wandb_run=self.wandb_run,
                    track_fidelity_losses=True
                )
                tr_loss = tr_res[0]
                tr_fid = tr_res[2] if len(tr_res) > 2 else {}
                
                self.train_losses.append(tr_loss)
                for p in self.combinatorial_patterns:
                    if tr_fid.get(p) is not None:
                        self.train_fidelity_losses[p].append(tr_fid[p])
                        
                # Validate
                self.model.eval()
                val_res = run_epoch(
                    self.val_loader, self.model, self.loss_fn,
                    self.optimizer, self.lr_scheduler, 10,
                    mode="eval", track_fidelity_losses=True
                )
                val_loss = val_res[0]
                val_fid = val_res[2] if len(val_res) > 2 else {}
                
                self.val_losses.append(val_loss)
                for p in self.combinatorial_patterns:
                    if val_fid.get(p) is not None:
                        self.val_fidelity_losses[p].append(val_fid[p])
                        
                # Logging (clean output, detailed tracking saved for plotting)
                print(f"Epoch {epoch+1}/{epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}")
                
                if self.wandb_run:
                    self.wandb_run.log({
                        "epoch": epoch,
                        "train_loss": tr_loss,
                        "val_loss": val_loss,
                        "best_val_loss": self.best_val_loss
                    })
                    
                # Save history (incremental)
                self.epochs_trained = epoch + 1
                self.save_history()
                
                # Early stopping - only save when finding new best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model to disk immediately (no GPU memory retention)
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    print(f"  ✓ New best model saved (Val Loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n🛑 Early stopping at epoch {epoch+1}")
                        break
                        
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            
        finally:
            # Restore best weights from checkpoint if it exists
            if self.checkpoint_path.exists():
                self.model.load_state_dict(
                    torch.load(self.checkpoint_path, map_location=self.config.device, weights_only=True)
                )
                print("✓ Best model weights restored from checkpoint")
                
            self.save_history()
            print(f"✓ Training state saved to: {self.history_path}")
            
        self.model.eval()
        print(f"\n✅ Training complete. Best Val Loss: {self.best_val_loss:.4f}")


# ============================================================================
# PART 5 COMPLETE
# ============================================================================


class ConformalCalibration:
    """Conformal Quantile Regression calibration with spatio-temporal stratification"""
    
    def __init__(self, config, model, pixelwise=True, temporal=False, n_seasons=4):
        self.config = config
        self.model = model
        self.target_coverage = 0.90
        self.alpha = 1.0 - self.target_coverage
        self.pixelwise = pixelwise  # Use pixelwise Q-scores for narrower intervals
        self.temporal = temporal  # Use temporal stratification (seasonal)
        self.n_seasons = n_seasons  # Number of temporal bins (default: 4 seasons)
        
        # Define scenarios for evaluation
        # New fidelity levels: Level 1 = 64 POD Ice Thickness, Level 2 = 64 POD SST, Level 3 = Sensors
        self.scenarios = {
            "3L_all": [False, False, False],                    # All three levels
            "2L_noSensors": [False, False, True],              # Thickness + SST (no Sensors)
            "2L_NoThickness": [True, False, False],            # SST + Sensors (no Thickness)
            "2L_no_SST": [False, True, False],                 # Thickness + Sensors (no SST)
            "1L_Sensors": [True, True, False],                 # Only Sensors
            "1L_SST": [True, False, True],                     # Only 64 POD SST
            "1L_Thickness": [False, True, True]                # Only 64 POD Ice Thickness
        }
        
    @staticmethod
    def day_to_season(day_of_year, n_seasons=4):
        """Map day of year (1-365) to season index (0 to n_seasons-1)
        
        Antarctic seasons (Southern Hemisphere):
        Season 0: Summer (Dec-Feb): days 335-365, 1-59
        Season 1: Autumn (Mar-May): days 60-151
        Season 2: Winter (Jun-Aug): days 152-243
        Season 3: Spring (Sep-Nov): days 244-334
        """
        if n_seasons == 4:
            # Antarctic seasonal boundaries
            if day_of_year >= 335 or day_of_year < 60:  # Dec 1 - Feb 28
                return 0  # Summer
            elif 60 <= day_of_year < 152:  # Mar 1 - May 31
                return 1  # Autumn
            elif 152 <= day_of_year < 244:  # Jun 1 - Aug 31
                return 2  # Winter
            else:  # 244 <= day_of_year < 335:  # Sep 1 - Nov 30
                return 3  # Spring
        else:
            # Generic: divide year into n_seasons equal bins
            return min(int(day_of_year * n_seasons / 365), n_seasons - 1)
    
    @staticmethod
    def force_mask(batch, mask_list, device):
        """Force a specific mask configuration on a batch"""
        B = batch['level_0'].shape[0]
        mask_tensor = torch.tensor(mask_list, dtype=torch.bool, device=device)
        batch['mask'] = mask_tensor.unsqueeze(0).expand(B, -1)
        return batch
        
    def calibrate_conditional(self, val_loader, residual_scaler, baseline_val, y_true_sic_val):
        """Calibrate Q-scores for each scenario
        
        CRITICAL: Theoretically correct conformal calibration:
        1. Denormalize predicted residuals (scaled_pred * std + mean)
        2. Add DMD baseline to get uncalibrated SIC predictions
        3. Clip to [0, 1] physical range
        4. Compute conformity scores on FINAL PHYSICAL SIC values
        
        This ensures coverage guarantees hold for the actual predictions (SIC),
        not intermediate residuals, and properly accounts for boundary clamping.
        
        Args:
            val_loader: Validation dataloader
            residual_scaler: Dict with 'mean' and 'std' for residual denormalization
            baseline_val: (n_val_samples, n_pixels) - DMD baseline for validation set
            y_true_sic_val: (n_val_samples, n_pixels) - True SIC values for validation
        """
        print("\n" + "="*70)
        print("CONFORMAL CALIBRATION")
        print("="*70)
        print(f"Target Coverage: {self.target_coverage:.0%}")
        
        method_desc = []
        if self.pixelwise:
            method_desc.append("Pixelwise")
        else:
            method_desc.append("Global")
        if self.temporal:
            method_desc.append(f"Temporal ({self.n_seasons} seasons)")
        print(f"Method: {' + '.join(method_desc)} Q-scores")
        
        self.model.eval()
        q_dict = {}
        
        # Diagnostics: track uncalibrated interval widths
        diagnostics = {'uncalibrated_widths': [], 'base_widths': []}
        
        # Extract scaling parameters
        res_mean = residual_scaler['mean']
        res_std = residual_scaler['std']
        
        for name, mask_cfg in self.scenarios.items():
            if self.temporal:
                # Spatio-temporal: dict of lists per season
                all_scores = {s: [] for s in range(self.n_seasons)}
            else:
                # Spatial only: single list
                all_scores = []
            
            # Diagnostics for this scenario
            uncal_widths = []
            
            # Use mixed precision for inference to save GPU memory
            use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
            
            batch_idx = 0
            curr_t = 0  # Track current time index for baseline and true SIC extraction
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Calibrating {name}"):
                    # Prepare inputs
                    batch_input = {k: v.to(self.config.device) for k, v in batch.items() if k != 'target'}
                    target = batch['target'].to(self.config.device)  # Normalized residuals
                    
                    # CRITICAL: Force scenario mask to ensure calibration matches evaluation
                    self.force_mask(batch_input, mask_cfg, self.config.device)
                    
                    # Forward pass with automatic mixed precision
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            out = self.model(batch_input)
                    else:
                        out = self.model(batch_input)
                    
                    # Reshape to (B, Seq, 3, Pixels)
                    B, S, _ = out.shape
                    n_pixels = out.shape[-1] // 3
                    preds = out.view(B, S, 3, n_pixels)
                    
                    # Extract quantiles (still normalized residuals) and move to CPU immediately
                    low_norm_cpu = preds[:, :, 0, :].cpu()  # Lower quantile (0.05)
                    high_norm_cpu = preds[:, :, 2, :].cpu()  # Upper quantile (0.95)
                    y_norm_cpu = target.cpu()
                    
                    # Free GPU tensors BEFORE creating new ones
                    del out, preds, batch_input, target
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Denormalize residuals on CPU (avoid GPU memory allocation)
                    low_res_cpu = low_norm_cpu * res_std + res_mean
                    high_res_cpu = high_norm_cpu * res_std + res_mean
                    y_res_cpu = y_norm_cpu * res_std + res_mean
                    
                    # Free normalized tensors
                    del low_norm_cpu, high_norm_cpu, y_norm_cpu
                    
                    # CRITICAL: Add DMD baseline and clamp to get FINAL SIC predictions
                    # This is the theoretically correct approach for conformal prediction
                    # Conformity scores must be computed on the actual predictions we report
                    
                    # Get DMD baseline for this batch
                    end_t = curr_t + B * S
                    base_slice = baseline_val[curr_t:end_t]  # (B*S, n_pixels)
                    base_np = base_slice.reshape(B, S, n_pixels)  # (B, S, n_pixels)
                    
                    # Get true SIC for this batch
                    y_true_slice = y_true_sic_val[curr_t:end_t]  # (B*S, n_pixels)
                    y_true_np = y_true_slice.reshape(B, S, n_pixels)  # (B, S, n_pixels)
                    
                    # Reconstruct FINAL SIC predictions with clamping
                    low_res_np = low_res_cpu.numpy()  # Shape: (B, S, n_pixels)
                    high_res_np = high_res_cpu.numpy()  # Shape: (B, S, n_pixels)
                    
                    sic_low_np = np.clip(base_np + low_res_np, 0, 1)
                    sic_high_np = np.clip(base_np + high_res_np, 0, 1)
                    
                    # Free temporary arrays
                    del low_res_cpu, high_res_cpu, y_res_cpu, low_res_np, high_res_np
                    
                    # DIAGNOSTIC: Compute uncalibrated interval width (in SIC space)
                    uncal_width = (sic_high_np - sic_low_np).mean()
                    uncal_widths.append(uncal_width)
                    
                    # Compute conformity scores on FINAL SIC values (theoretically correct)
                    # Score per pixel: max(sic_low - sic_true, sic_true - sic_high)
                    scores_np = np.maximum(sic_low_np - y_true_np, y_true_np - sic_high_np)  # Shape: (B, S, n_pixels)
                    
                    # Free SIC arrays
                    del sic_low_np, sic_high_np, base_np, y_true_np
                    
                    # Update time counter for next batch
                    curr_t = end_t
                    
                    # Compute temporal indices if needed
                    if self.temporal:
                        # Compute day of year for each sample in batch
                        # Each batch has B samples, S sequence steps
                        # Assume validation data is organized sequentially (days 0-364 repeating per year)
                        # Total samples processed so far: batch_idx * batch_size
                        # Each sample in batch has index: (batch_idx * B) + b
                        
                        for b in range(B):
                            for s in range(S):
                                # Compute cumulative sample index
                                cumulative_idx = batch_idx * B + b + s
                                # Map to day of year (0-364, then convert to 1-365)
                                day_idx = cumulative_idx % 365
                                season_idx = self.day_to_season(day_idx + 1, self.n_seasons)  # day_of_year is 1-indexed
                                
                                # Extract scores for this specific time step
                                if self.pixelwise:
                                    score_sample = scores_np[b, s, :]  # (n_pixels,)
                                    all_scores[season_idx].append(score_sample)
                                else:
                                    score_sample = scores_np[b, s, :].flatten()
                                    all_scores[season_idx].append(score_sample)
                    else:
                        # No temporal stratification
                        if self.pixelwise:
                            # Reshape to (B*S, n_pixels) to maintain pixel correspondence
                            scores_reshaped = scores_np.reshape(-1, n_pixels)
                            all_scores.append(scores_reshaped)
                        else:
                            # Flatten all for global max
                            all_scores.append(scores_np.flatten())
                    
                    batch_idx += 1
                    
                    # Clean up
                    del scores_np
                    
            # Store diagnostics
            avg_uncal_width = np.mean(uncal_widths)
            diagnostics['uncalibrated_widths'].append(avg_uncal_width)
            
            # Compute Q-hat (quantile)
            if self.temporal:
                # Spatio-temporal: compute Q-scores per (pixel, season)
                season_names = ['Summer', 'Autumn', 'Winter', 'Spring']
                
                if self.pixelwise:
                    # Shape: (n_pixels, n_seasons)
                    q_per_pixel_season = np.zeros((n_pixels, self.n_seasons))
                    
                    for s in range(self.n_seasons):
                        if len(all_scores[s]) > 0:
                            scores_season = np.stack(all_scores[s], axis=0)  # (N_season, n_pixels)
                            q_per_pixel_season[:, s] = np.quantile(scores_season, 1 - self.alpha, axis=0, method='higher')
                        else:
                            print(f"    WARNING: No samples in season {s}, using zeros")
                            q_per_pixel_season[:, s] = 0.0
                    
                    q_hat = q_per_pixel_season  # Store as (n_pixels, n_seasons) array
                    
                    # Print detailed statistics
                    print(f"  {name:12s} → Spatio-Temporal Q-Scores:")
                    for s in range(self.n_seasons):
                        q_season = q_per_pixel_season[:, s]
                        s_name = season_names[s] if s < len(season_names) else f"Season{s}"
                        n_samples = len(all_scores[s])
                        print(f"    {s_name:8s}: mean={q_season.mean():.5f}, std={q_season.std():.5f}, "
                              f"range=[{q_season.min():.5f}, {q_season.max():.5f}], n={n_samples}")
                    
                    # Overall statistics
                    q_overall_mean = q_per_pixel_season.mean()
                    print(f"    {'Overall':8s}: mean={q_overall_mean:.5f}")
                else:
                    # Global per season: (n_seasons,) array
                    q_per_season = np.zeros(self.n_seasons)
                    
                    for s in range(self.n_seasons):
                        if len(all_scores[s]) > 0:
                            scores_season = np.concatenate(all_scores[s])
                            q_per_season[s] = np.quantile(scores_season, 1 - self.alpha, method='higher')
                        else:
                            print(f"    WARNING: No samples in season {s}")
                            q_per_season[s] = 0.0
                    
                    q_hat = q_per_season
                    
                    print(f"  {name:12s} → Global Q-Scores by Season:")
                    for s in range(self.n_seasons):
                        s_name = season_names[s] if s < len(season_names) else f"Season{s}"
                        print(f"    {s_name:8s}: Q={q_per_season[s]:.5f}")
            else:
                # No temporal stratification (original code)
                if self.pixelwise:
                    # Pixelwise: compute Q-score per pixel
                    all_scores = np.concatenate(all_scores, axis=0)  # (N_total, n_pixels)
                    q_per_pixel = np.quantile(all_scores, 1 - self.alpha, axis=0, method='higher')  # (n_pixels,)
                    q_hat = q_per_pixel  # Store pixelwise Q-scores
                    q_mean = q_per_pixel.mean()
                    q_std = q_per_pixel.std()
                    print(f"  {name:12s} → Pixelwise Q-Score: mean={q_mean:.5f}, std={q_std:.5f}, range=[{q_per_pixel.min():.5f}, {q_per_pixel.max():.5f}]")
                else:
                    # Global: single Q-score from max over all pixels
                    all_scores = np.concatenate(all_scores)
                    q_hat = np.quantile(all_scores, 1 - self.alpha, method='higher')
                    print(f"  {name:12s} → Global Q-Score: {q_hat:.5f}")
            
            # DIAGNOSTIC: Print uncalibrated vs calibrated width comparison
            if isinstance(q_hat, np.ndarray):
                q_mean_val = q_hat.mean()
            else:
                q_mean_val = q_hat
            expected_cal_width = avg_uncal_width + 2 * q_mean_val
            print(f"    📊 Width: Uncal={avg_uncal_width:.4f}, +2Q={expected_cal_width:.4f}, Increase={(expected_cal_width/avg_uncal_width - 1)*100:.1f}%")
            
            q_dict[name] = q_hat
        
        # Store diagnostics
        self.diagnostics = diagnostics
        
        print("✓ Calibration complete")
        print("  Note: Q-scores computed on FINAL SIC predictions (theoretically correct approach)")
        print("        This ensures coverage guarantees hold for actual physical SIC values")
        print(f"\n📊 CALIBRATION DIAGNOSTICS SUMMARY:")
        print(f"  Average uncalibrated interval width (SIC space): {np.mean(diagnostics['uncalibrated_widths']):.4f}")
        print(f"  This is the model's base uncertainty in SIC space before conformal correction")
        
        return q_dict


class TestEvaluator:
    """Evaluate model on test set with physical metrics"""
    
    def __init__(self, config, model, q_scores, scenarios, residual_scaler):
        self.config = config
        self.model = model
        self.q_scores = q_scores
        self.scenarios = scenarios
        self.res_mean = residual_scaler['mean']
        self.res_std = residual_scaler['std']
        
    @staticmethod
    def day_to_season(day_of_year, n_seasons=4):
        """Map day of year to season (same as in ConformalCalibration)"""
        if n_seasons == 4:
            if day_of_year >= 335 or day_of_year < 60:
                return 0  # Summer
            elif 60 <= day_of_year < 152:
                return 1  # Autumn
            elif 152 <= day_of_year < 244:
                return 2  # Winter
            else:
                return 3  # Spring
        else:
            return min(int(day_of_year * n_seasons / 365), n_seasons - 1)
    
    @staticmethod
    def force_mask(batch, mask_list, device):
        """Force a specific mask configuration"""
        B = batch['level_0'].shape[0]
        mask_tensor = torch.tensor(mask_list, dtype=torch.bool, device=device)
        batch['mask'] = mask_tensor.unsqueeze(0).expand(B, -1)
        return batch
        
    def evaluate_test_physics(self, test_loader, dmd_baseline, region_mask, ground_truth_sic):
        """
        Evaluate on test set comparing Hybrid (DMD + Model) vs DMD baseline
        
        Args:
            test_loader: Test dataloader
            dmd_baseline: (TotalTimeTest, nx*ny) - DMD predictions on full grid
            region_mask: (ny, nx) - Boolean mask for active region
            ground_truth_sic: (TotalTimeTest, n_pixels_active) - Original true SIC from .pkl file
        """
        print("\n" + "="*70)
        print("TEST EVALUATION - PHYSICAL METRICS")
        print("="*70)
        
        self.model.eval()
        results = []
        mask_flat = region_mask.reshape(-1)
        
        season_names = ['Summer', 'Autumn', 'Winter', 'Spring']
        
        for name, mask_cfg in self.scenarios.items():
            q = self.q_scores[name]
            
            # Determine Q-score type
            if isinstance(q, np.ndarray) and q.ndim == 2:
                # Spatio-temporal: (n_pixels, n_seasons)
                is_spatiotemporal = True
                is_pixelwise = True
                q_tensor = torch.tensor(q, dtype=torch.float32)  # (n_pixels, n_seasons)
                print(f"  Using spatio-temporal Q-scores for {name}: shape {q.shape}")
                print(f"    Seasonal means: {[f'{season_names[i]}={q[:, i].mean():.5f}' for i in range(min(4, q.shape[1]))]}")
            elif isinstance(q, np.ndarray) and q.ndim == 1:
                # Check if it's temporal-only or pixelwise-only
                if len(q) <= 12:  # Likely temporal (seasons or months)
                    is_spatiotemporal = True
                    is_pixelwise = False
                    q_tensor = torch.tensor(q, dtype=torch.float32)  # (n_seasons,)
                    print(f"  Using temporal Q-scores for {name}: {[f'{season_names[i] if i < 4 else i}={q[i]:.5f}' for i in range(len(q))]}")
                else:  # Pixelwise
                    is_spatiotemporal = False
                    is_pixelwise = True
                    q_tensor = torch.tensor(q, dtype=torch.float32).view(1, 1, -1)
                    print(f"  Using pixelwise Q-scores for {name}: shape {q.shape}, mean={q.mean():.5f}, std={q.std():.5f}")
            else:
                # Global Q-score: single value
                is_spatiotemporal = False
                is_pixelwise = False
                q_tensor = torch.tensor(q, dtype=torch.float32)
                print(f"  Using global Q-score for {name}: {q:.5f}")
            
            total_mae_model = 0
            total_mae_dmd = 0
            covered_count = 0
            total_count = 0
            total_width = 0
            
            current_time_idx = 0
            batch_idx = 0
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Evaluating {name}"):
                    # Predict residuals
                    batch_input = {k: v.to(self.config.device) for k, v in batch.items() if k != 'target'}
                    self.force_mask(batch_input, mask_cfg, self.config.device)
                    out = self.model(batch_input)
                    
                    B, S, _ = out.shape
                    n_pixels = out.shape[-1] // 3
                    
                    # Move to CPU and reshape immediately
                    out_cpu = out.cpu()
                    del out  # Free GPU memory
                    
                    # Reshape predictions
                    out_reshaped = out_cpu.view(B, S, 3, n_pixels)
                    
                    # Apply Q-scores (depends on type)
                    if is_spatiotemporal:
                        # Need to apply season-specific Q-scores
                        res_low = out_reshaped[:, :, 0, :].clone()
                        res_high = out_reshaped[:, :, 2, :].clone()
                        
                        # Compute season for each (b, s) in batch
                        for b in range(B):
                            for s in range(S):
                                day_idx = (current_time_idx + s) % 365
                                season_idx = self.day_to_season(day_idx + 1, q_tensor.shape[-1] if q_tensor.ndim > 0 else 4)
                                
                                if is_pixelwise:
                                    # q_tensor shape: (n_pixels, n_seasons)
                                    q_values = q_tensor[:, season_idx]  # (n_pixels,)
                                    res_low[b, s, :] -= q_values
                                    res_high[b, s, :] += q_values
                                else:
                                    # q_tensor shape: (n_seasons,)
                                    q_value = q_tensor[season_idx].item()
                                    res_low[b, s, :] -= q_value
                                    res_high[b, s, :] += q_value
                    elif is_pixelwise:
                        # Pixelwise but not temporal: q_tensor shape (1, 1, n_pixels)
                        res_low = out_reshaped[:, :, 0, :] - q_tensor
                        res_high = out_reshaped[:, :, 2, :] + q_tensor
                    else:
                        # Global scalar
                        res_low = out_reshaped[:, :, 0, :] - q_tensor
                        res_high = out_reshaped[:, :, 2, :] + q_tensor
                    
                    res_med = out_reshaped[:, :, 1, :]
                    del out_cpu, out_reshaped  # Free memory
                    
                    # CRITICAL FIX: Denormalize predicted residuals
                    # Model outputs are in normalized space (mean=0, std=1)
                    # Must denormalize before adding to DMD baseline
                    res_low_denorm = res_low * self.res_std + self.res_mean
                    res_med_denorm = res_med * self.res_std + self.res_mean
                    res_high_denorm = res_high * self.res_std + self.res_mean
                    del res_low, res_med, res_high  # Free normalized versions
                    
                    batch_idx += 1
                    
                    # Get DMD baseline for this time chunk
                    # CRITICAL: Handle batch dimension properly - get B*S days not just S days
                    end_idx = current_time_idx + B * S
                    base_chunk = dmd_baseline[current_time_idx:end_idx, mask_flat]  # (B*S, n_pixels)
                    base_tensor = torch.tensor(base_chunk, dtype=torch.float32).view(B, S, -1)  # (B, S, n_pixels)
                    
                    # Get ACTUAL ground truth SIC from original data (no reconstruction!)
                    sic_true_chunk = ground_truth_sic[current_time_idx:end_idx]  # (B*S, n_pixels)
                    sic_true = sic_true_chunk.view(B, S, -1)  # (B, S, n_pixels)
                    
                    # Reconstruct model predictions from DMD baseline + denormalized residuals (clamp to [0, 1])
                    sic_pred = torch.clamp(base_tensor + res_med_denorm, 0, 1)
                    sic_low = torch.clamp(base_tensor + res_low_denorm, 0, 1)
                    sic_high = torch.clamp(base_tensor + res_high_denorm, 0, 1)
                    sic_dmd = torch.clamp(base_tensor, 0, 1)
                    
                    # Compute metrics
                    total_mae_model += torch.abs(sic_true - sic_pred).sum().item()
                    total_mae_dmd += torch.abs(sic_true - sic_dmd).sum().item()
                    
                    # Coverage
                    is_covered = (sic_true >= sic_low) & (sic_true <= sic_high)
                    covered_count += is_covered.sum().item()
                    total_count += is_covered.numel()
                    total_width += (sic_high - sic_low).sum().item()
                    
                    # Update time index by B*S (total days processed in this batch)
                    current_time_idx += B * S
                    
                    # Clean up batch memory
                    del batch_input, res_low_denorm, res_med_denorm, res_high_denorm
                    del base_tensor, sic_pred, sic_low, sic_high, sic_true, sic_dmd, is_covered
                    torch.cuda.empty_cache()
                    
            # Aggregate metrics
            mae_model = total_mae_model / total_count
            mae_dmd = total_mae_dmd / total_count
            improvement = (mae_dmd - mae_model) / (mae_dmd + 1e-6) * 100
            coverage = covered_count / total_count
            width = total_width / total_count
            
            # DIAGNOSTIC: Print detailed stats
            print(f"  {name:12s} → MAE_Model: {mae_model:.6f}, MAE_DMD: {mae_dmd:.6f}")
            print(f"               Improvement: {improvement:+.2f}% | Coverage: {coverage:.1%} | Width: {width:.4f}")
            print(f"               Mask config: {mask_cfg} | Samples: {total_count}")
            
            results.append({
                "Scenario": name,
                "MAE_Ours": mae_model,
                "MAE_DMD": mae_dmd,
                "Improvement_%": improvement,
                "Coverage": coverage,
                "Width": width
            })
            
            print(f"  {name:12s} → MAE: {mae_model:.4f} (DMD: {mae_dmd:.4f}) | "
                  f"Improvement: {improvement:+.1f}% | Coverage: {coverage:.1%}")
                  
        return pd.DataFrame(results)
        
    def generate_full_sic_predictions(self, test_loader, baseline_masked, q_score, y_true_sic, mask_config=None):
        """
        Generate full SIC predictions for visualization
        
        CRITICAL: Q-scores are now in SIC space (not residual space).
        Order of operations:
        1. Predict residuals (normalized)
        2. Denormalize residuals
        3. Add DMD baseline and clamp to get uncalibrated SIC
        4. Apply Q-scores in SIC space: sic_low - Q, sic_high + Q
        5. Final clamp to [0, 1]
        
        Args:
            test_loader: Test dataloader
            baseline_masked: (TotalTime, n_pixels_region) - DMD baseline on active region
            q_score: Float, array(n_pixels), or array(n_pixels, n_seasons) - Calibration correction IN SIC SPACE
            y_true_sic: (TotalTime, n_pixels_region) - Original true SIC values (not residuals)
            mask_config: List of 3 bools [mask_level1, mask_level2, mask_level3] - if None, uses dataset default
            
        Returns:
            Tuple of (low, median, high, true) predictions
        """
        print("\nGENERATING FULL SIC PREDICTIONS")
        print("-"*70)
        
        if mask_config is not None:
            print(f"  Using FORCED mask configuration: {mask_config}")
        else:
            print("  Using dataset default masks (random during training)")
        
        # Determine Q-score type
        is_spatiotemporal = isinstance(q_score, np.ndarray) and q_score.ndim == 2
        is_pixelwise = isinstance(q_score, np.ndarray) and q_score.ndim == 1
        
        if is_spatiotemporal:
            print(f"  Using spatio-temporal Q-scores: shape {q_score.shape}")
        elif is_pixelwise:
            print(f"  Using pixelwise Q-scores: shape {q_score.shape}")
        else:
            print(f"  Using global Q-score: {q_score}")
        
        self.model.eval()
        l_low, l_med, l_high, l_true = [], [], [], []
        curr_t = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                # Input
                batch_input = {k: v.to(self.config.device) for k, v in batch.items() if k != 'target'}
                
                # CRITICAL FIX: Force mask configuration if specified
                if mask_config is not None:
                    self.force_mask(batch_input, mask_config, self.config.device)
                
                # Predict
                out = self.model(batch_input)
                B, S, _ = out.shape
                n_pixels = out.shape[-1] // 3
                
                # FREE GPU: Move to CPU immediately and free GPU tensors
                out_cpu = out.cpu()
                del out, batch_input
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Reshape predictions (now on CPU)
                preds = out_cpu.view(B, S, 3, n_pixels)
                del out_cpu
                
                # Extract quantiles (still normalized residuals)
                r_low_norm = preds[:, :, 0, :]
                r_med_norm = preds[:, :, 1, :]
                r_high_norm = preds[:, :, 2, :]
                del preds
                
                # Step 1: Denormalize predicted residuals
                r_low_denorm = r_low_norm * self.res_std + self.res_mean
                r_med_denorm = r_med_norm * self.res_std + self.res_mean
                r_high_denorm = r_high_norm * self.res_std + self.res_mean
                del r_low_norm, r_med_norm, r_high_norm
                
                # Step 2: Get baseline (keep on CPU to avoid GPU memory)
                end_t = curr_t + B * S
                base_slice = baseline_masked[curr_t:end_t]  # (B*S, n_pixels)
                base_tensor = base_slice.view(B, S, -1)  # (B, S, n_pixels)
                
                # Get original true SIC (keep on CPU)
                sic_true_slice = y_true_sic[curr_t:end_t]  # (B*S, n_pixels)
                sic_true_tensor = sic_true_slice.view(B, S, -1)  # (B, S, n_pixels)
                
                # Step 3: Reconstruct UNCALIBRATED SIC (add baseline, clamp to [0,1])
                sic_med_uncal = torch.clamp(base_tensor + r_med_denorm, 0, 1)
                sic_low_uncal = torch.clamp(base_tensor + r_low_denorm, 0, 1)
                sic_high_uncal = torch.clamp(base_tensor + r_high_denorm, 0, 1)
                del r_low_denorm, r_med_denorm, r_high_denorm
                
                # Step 4: Apply Q-scores in SIC space (Q-scores are in SIC units, not residual units)
                if is_spatiotemporal:
                    # Q-score shape: (n_pixels, n_seasons)
                    q_tensor = torch.tensor(q_score, dtype=sic_low_uncal.dtype, device='cpu')
                    sic_low_cal = sic_low_uncal.clone()
                    sic_high_cal = sic_high_uncal.clone()
                    
                    # Apply season-specific Q-scores
                    for b in range(B):
                        for s in range(S):
                            day_idx = (curr_t + b * S + s) % 365
                            season_idx = self.day_to_season(day_idx + 1, q_tensor.shape[-1])
                            q_values = q_tensor[:, season_idx]  # (n_pixels,)
                            sic_low_cal[b, s, :] -= q_values
                            sic_high_cal[b, s, :] += q_values
                elif is_pixelwise:
                    # Q-score shape: (n_pixels,)
                    q_tensor = torch.tensor(q_score, dtype=sic_low_uncal.dtype, device='cpu').view(1, 1, -1)
                    sic_low_cal = sic_low_uncal - q_tensor
                    sic_high_cal = sic_high_uncal + q_tensor
                else:
                    # Global scalar Q-score
                    q_tensor = torch.tensor(q_score, dtype=sic_low_uncal.dtype, device='cpu')
                    sic_low_cal = sic_low_uncal - q_tensor
                    sic_high_cal = sic_high_uncal + q_tensor
                
                del sic_low_uncal, sic_high_uncal
                
                # Step 5: Final clamp to ensure [0, 1] range
                s_low = torch.clamp(sic_low_cal, 0, 1)
                s_high = torch.clamp(sic_high_cal, 0, 1)
                s_med = sic_med_uncal  # Median doesn't get Q-score adjustment
                s_true = torch.clamp(sic_true_tensor, 0, 1)
                del sic_low_cal, sic_high_cal, sic_med_uncal
                
                # Accumulate (already on CPU)
                l_med.append(s_med)
                l_low.append(s_low)
                l_high.append(s_high)
                l_true.append(s_true)
                
                # Update time index by B*S (total days processed in this batch)
                curr_t += B * S
                
                # Clean up CPU tensors
                del base_slice, base_tensor, sic_true_slice, sic_true_tensor
                del s_low, s_high, s_med, s_true
                # GPU already cleaned earlier
                
        print("✓ Predictions generated")
        
        # Final safety clipping to ensure [0,1] constraints (probabilities)
        return (
            torch.clamp(torch.cat(l_low, dim=1), 0, 1),
            torch.clamp(torch.cat(l_med, dim=1), 0, 1),
            torch.clamp(torch.cat(l_high, dim=1), 0, 1),
            torch.clamp(torch.cat(l_true, dim=1), 0, 1)
        )


# ============================================================================
# PART 6 COMPLETE
# ============================================================================


# ============================================================================
# ICENET-STYLE METRICS FOR COMPARISON
# ============================================================================

class IceNetMetrics:
    """
    IceNet-compatible metrics for sea ice forecasting benchmarking.
    
    This class implements:
    1. IIEE (Integrated Ice-Edge Error) and Binary Accuracy
    2. Mean Prediction Interval Width (MPIW) for Marginal Ice Zone (MIZ)
    3. Calibration/Reliability diagrams
    
    References:
    - Andersson et al. (2021) "Seasonal Arctic sea ice forecasting with probabilistic deep learning"
    - IceNet paper: https://www.nature.com/articles/s41467-021-25257-4
    """
    
    def __init__(self, grid_cell_area_km2=None):
        """
        Args:
            grid_cell_area_km2: Area of each grid cell in km². 
                               If None, will be calculated assuming 25km resolution.
        """
        # Default: NSIDC 25km grid → each cell is 25×25 = 625 km²
        self.grid_cell_area = grid_cell_area_km2 if grid_cell_area_km2 is not None else 625.0
        
    def calculate_iiee(self, pred_sic, true_sic, ice_threshold=0.15):
        """
        Calculate Integrated Ice-Edge Error (IIEE).
        
        IIEE measures the total area (km²) where the ice edge is misplaced.
        Ice edge is defined as the 15% SIC contour.
        
        Args:
            pred_sic: (T, n_pixels) or (n_pixels,) - Predicted SIC [0, 1]
            true_sic: (T, n_pixels) or (n_pixels,) - True SIC [0, 1]
            ice_threshold: Ice edge threshold (default: 0.15 = 15%)
            
        Returns:
            iiee: Total misplacement area in km²
            overestimation: Area where model predicts ice but truth is open water (km²)
            underestimation: Area where truth is ice but model predicts open water (km²)
        """
        # Ensure numpy arrays
        if torch.is_tensor(pred_sic):
            pred_sic = pred_sic.cpu().numpy()
        if torch.is_tensor(true_sic):
            true_sic = true_sic.cpu().numpy()
            
        # Binarize at ice edge threshold
        pred_ice = (pred_sic > ice_threshold).astype(float)
        true_ice = (true_sic > ice_threshold).astype(float)
        
        # Calculate errors
        overestimation_mask = (pred_ice > true_ice)  # False positive ice
        underestimation_mask = (true_ice > pred_ice)  # False negative ice
        
        # Sum over all pixels and timesteps, convert to km²
        O = np.sum(overestimation_mask) * self.grid_cell_area
        U = np.sum(underestimation_mask) * self.grid_cell_area
        
        iiee = O + U
        
        return iiee, O, U
    
    def calculate_binary_accuracy(self, pred_sic, true_sic, active_mask=None, ice_threshold=0.15):
        """
        Calculate Binary Accuracy (IceNet-style).
        
        Binary Accuracy = (1 - IIEE / Active_Grid_Area) × 100%
        
        Args:
            pred_sic: (T, n_pixels) or (n_pixels,) - Predicted SIC
            true_sic: (T, n_pixels) or (n_pixels,) - True SIC
            active_mask: (n_pixels,) - Boolean mask of active region (or None to use all)
            ice_threshold: Ice edge threshold (default: 15%)
            
        Returns:
            binary_accuracy: Percentage [0, 100]
            iiee: Total error in km²
            total_area: Active grid area in km²
        """
        # Calculate IIEE
        iiee, O, U = self.calculate_iiee(pred_sic, true_sic, ice_threshold)
        
        # Calculate total active area
        if active_mask is not None:
            n_active_pixels = np.sum(active_mask)
        else:
            # Use all pixels
            if pred_sic.ndim == 1:
                n_active_pixels = len(pred_sic)
            else:
                n_active_pixels = pred_sic.shape[1]
        
        # Account for temporal dimension if present
        if pred_sic.ndim == 2:
            n_timesteps = pred_sic.shape[0]
            total_area = n_active_pixels * n_timesteps * self.grid_cell_area
        else:
            total_area = n_active_pixels * self.grid_cell_area
        
        # Binary accuracy
        binary_accuracy = (1 - iiee / total_area) * 100.0
        
        return binary_accuracy, iiee, total_area, O, U
    
    def calculate_miz_ci_width(self, pred_low, pred_high, true_sic, 
                               miz_lower=0.15, miz_upper=0.80):
        """
        Calculate Mean Prediction Interval Width (MPIW) in the Marginal Ice Zone.
        
        The MIZ is defined as regions where 15% < SIC < 80%.
        
        Args:
            pred_low: (T, n_pixels) - Lower bound of prediction interval
            pred_high: (T, n_pixels) - Upper bound of prediction interval
            true_sic: (T, n_pixels) - True SIC to identify MIZ
            miz_lower: Lower MIZ threshold (default: 15%)
            miz_upper: Upper MIZ threshold (default: 80%)
            
        Returns:
            mpiw_miz: Mean width in MIZ
            mpiw_all: Mean width over all pixels
            miz_fraction: Fraction of pixels in MIZ
        """
        # Ensure numpy
        if torch.is_tensor(pred_low):
            pred_low = pred_low.cpu().numpy()
        if torch.is_tensor(pred_high):
            pred_high = pred_high.cpu().numpy()
        if torch.is_tensor(true_sic):
            true_sic = true_sic.cpu().numpy()
        
        # Calculate widths
        widths = pred_high - pred_low
        
        # Identify MIZ pixels
        miz_mask = (true_sic > miz_lower) & (true_sic < miz_upper)
        
        # Calculate metrics
        if np.sum(miz_mask) > 0:
            mpiw_miz = np.mean(widths[miz_mask])
            miz_fraction = np.sum(miz_mask) / miz_mask.size
        else:
            mpiw_miz = np.nan
            miz_fraction = 0.0
        
        mpiw_all = np.mean(widths)
        
        return mpiw_miz, mpiw_all, miz_fraction
    
    def calculate_reliability_diagram_data(self, pred_low, pred_high, true_sic, n_bins=10):
        """
        Calculate data for calibration/reliability diagram.
        
        This verifies that a 90% confidence interval actually contains the true value
        90% of the time.
        
        Args:
            pred_low: (T, n_pixels) - Lower bound (e.g., 5th percentile)
            pred_high: (T, n_pixels) - Upper bound (e.g., 95th percentile)
            true_sic: (T, n_pixels) - True SIC values
            n_bins: Number of bins for stratification by prediction uncertainty
            
        Returns:
            coverage_overall: Overall coverage fraction
            bin_centers: Bin center values (mean width per bin)
            bin_coverage: Empirical coverage in each bin
            bin_counts: Number of samples in each bin
        """
        # Ensure numpy
        if torch.is_tensor(pred_low):
            pred_low = pred_low.cpu().numpy()
        if torch.is_tensor(pred_high):
            pred_high = pred_high.cpu().numpy()
        if torch.is_tensor(true_sic):
            true_sic = true_sic.cpu().numpy()
        
        # Flatten all arrays
        pred_low_flat = pred_low.flatten()
        pred_high_flat = pred_high.flatten()
        true_sic_flat = true_sic.flatten()
        
        # Calculate coverage (is true value inside interval?)
        is_covered = (true_sic_flat >= pred_low_flat) & (true_sic_flat <= pred_high_flat)
        coverage_overall = np.mean(is_covered)
        
        # Calculate widths for binning
        widths = pred_high_flat - pred_low_flat
        
        # Create bins based on width (uncertainty stratification)
        # Bins go from min to max width
        width_min, width_max = np.percentile(widths, [1, 99])  # Exclude extreme outliers
        bins = np.linspace(width_min, width_max, n_bins + 1)
        
        # Initialize storage
        bin_centers = []
        bin_coverage = []
        bin_counts = []
        
        # Calculate coverage per bin
        for i in range(n_bins):
            bin_mask = (widths >= bins[i]) & (widths < bins[i+1])
            
            if np.sum(bin_mask) > 0:
                bin_centers.append(np.mean(widths[bin_mask]))
                bin_coverage.append(np.mean(is_covered[bin_mask]))
                bin_counts.append(np.sum(bin_mask))
            else:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                bin_coverage.append(np.nan)
                bin_counts.append(0)
        
        return coverage_overall, np.array(bin_centers), np.array(bin_coverage), np.array(bin_counts)
    
    def compute_seasonal_iiee_binary_accuracy(self, pred_sic, true_sic, 
                                              days_per_year=365, n_seasons=4,
                                              ice_threshold=0.15, active_mask=None):
        """
        Calculate IIEE and Binary Accuracy for each season.
        
        Args:
            pred_sic: (T, n_pixels) - Predictions
            true_sic: (T, n_pixels) - Ground truth
            days_per_year: Number of days per year (default: 365)
            n_seasons: Number of seasons (default: 4 for quarterly seasons)
            ice_threshold: Ice edge threshold
            active_mask: Boolean mask for active pixels
            
        Returns:
            seasonal_metrics: Dict with keys for each season containing 
                            {'binary_accuracy', 'iiee', 'O', 'U'}
        """
        # Ensure numpy
        if torch.is_tensor(pred_sic):
            pred_sic = pred_sic.cpu().numpy()
        if torch.is_tensor(true_sic):
            true_sic = true_sic.cpu().numpy()
        
        n_timesteps = pred_sic.shape[0]
        days_per_season = days_per_year // n_seasons
        
        seasonal_metrics = {}
        season_names = ['Summer', 'Autumn', 'Winter', 'Spring'] if n_seasons == 4 else [f'Season_{i+1}' for i in range(n_seasons)]
        
        for season_idx in range(n_seasons):
            # Get days belonging to this season across all years
            season_mask = np.zeros(n_timesteps, dtype=bool)
            
            for year_start in range(0, n_timesteps, days_per_year):
                season_start = year_start + season_idx * days_per_season
                season_end = min(season_start + days_per_season, n_timesteps)
                if season_start < n_timesteps:
                    season_mask[season_start:season_end] = True
            
            # Extract seasonal data
            pred_season = pred_sic[season_mask]
            true_season = true_sic[season_mask]
            
            # Calculate metrics
            if len(pred_season) > 0:
                binary_acc, iiee, total_area, O, U = self.calculate_binary_accuracy(
                    pred_season, true_season, active_mask, ice_threshold
                )
                
                seasonal_metrics[season_names[season_idx]] = {
                    'binary_accuracy': binary_acc,
                    'iiee': iiee,
                    'total_area': total_area,
                    'overestimation': O,
                    'underestimation': U,
                    'n_timesteps': len(pred_season)
                }
        
        return seasonal_metrics
    
    def plot_reliability_diagram(self, coverage_overall, bin_centers, bin_coverage, 
                                 bin_counts, target_coverage=0.90, save_path=None):
        """
        Create a reliability diagram for calibration assessment.
        
        Args:
            coverage_overall: Overall empirical coverage
            bin_centers: Width bin centers
            bin_coverage: Coverage in each bin
            bin_counts: Sample counts per bin
            target_coverage: Nominal coverage level (e.g., 0.90 for 90% CI)
            save_path: Path to save figure (or None to return figure)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Reliability diagram (coverage vs uncertainty)
        valid_bins = ~np.isnan(bin_coverage)
        
        ax1.scatter(bin_centers[valid_bins], bin_coverage[valid_bins], 
                   s=100, alpha=0.7, c='steelblue', edgecolors='black')
        ax1.axhline(y=target_coverage, color='red', linestyle='--', linewidth=2,
                   label=f'Target Coverage ({target_coverage*100:.0f}%)')
        ax1.axhline(y=coverage_overall, color='green', linestyle='-', linewidth=2,
                   label=f'Overall Coverage ({coverage_overall*100:.1f}%)')
        
        ax1.set_xlabel('Prediction Interval Width', fontsize=12)
        ax1.set_ylabel('Empirical Coverage', fontsize=12)
        ax1.set_title('Calibration Reliability Diagram', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Right: Histogram of samples per bin
        ax2.bar(bin_centers[valid_bins], bin_counts[valid_bins], 
               width=np.diff(bin_centers[valid_bins]).mean() if len(bin_centers[valid_bins]) > 1 else 0.01,
               alpha=0.7, color='coral', edgecolor='black')
        ax2.set_xlabel('Prediction Interval Width', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Sample Distribution by Uncertainty', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved reliability diagram to {save_path}")
        else:
            return fig
    
    def plot_ci_width_map(self, ci_widths, mask_land, x, y, season_name="", save_path=None):
        """
        Plot spatial map of confidence interval widths.
        
        Args:
            ci_widths: (ny, nx) - CI width for each pixel
            mask_land: (ny, nx) - Land mask
            x, y: 2D coordinate arrays
            season_name: Season identifier for title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Mask land and zero widths
        ci_widths_masked = np.ma.masked_where(mask_land | (ci_widths == 0), ci_widths)
        
        # Plot
        im = ax.pcolormesh(x, y, ci_widths_masked, cmap='plasma', shading='auto',
                          vmin=0, vmax=np.percentile(ci_widths[~mask_land], 95))
        
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
        cbar.set_label('CI Width (SIC fraction)', fontsize=12)
        
        ax.set_title(f'90% Confidence Interval Width - {season_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (km)', fontsize=12)
        ax.set_ylabel('Y (km)', fontsize=12)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved CI width map to {save_path}")
        else:
            return fig


class ResultsSaver:
    """Save all numerical and visual results"""
    
    def __init__(self, config):
        self.config = config
        self.results_dir = config.results_dir / config.experiment
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Results will be saved to: {self.results_dir}")
        
    def save_training_curves(self, trainer):
        """Save training and validation loss curves"""
        print("\nSaving training curves...")
        
        eps = 1e-8
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # axes[0, 0]: Final bar chart (training losses only)
        all_patterns = ["3L_all", "2L_1+2", "2L_1+3", "2L_2+3", "1L_1", "1L_2", "1L_3"]
        final_train = [max(eps, trainer.train_fidelity_losses[p][-1]) if trainer.train_fidelity_losses[p] else eps for p in all_patterns]
        
        x_pos = np.arange(len(all_patterns))
        axes[0, 0].bar(x_pos, final_train, alpha=0.8, color='steelblue')
        axes[0, 0].set_xlabel('Fidelity Pattern')
        axes[0, 0].set_ylabel('Loss (log scale)')
        axes[0, 0].set_title('Final Training Loss by Pattern')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(all_patterns, rotation=45, ha='right')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, which='both', alpha=0.3)
        
        # axes[0, 1]: Single-level + Three-level training (with 3L_all emphasis)
        single_plus_three = ['1L_1', '1L_2', '1L_3', '3L_all']
        colors_single_three = {'1L_1': 'green', '1L_2': 'orange', '1L_3': 'purple', '3L_all': 'red'}
        axes[0, 1].set_title('Single & Three Level (Train)')
        for pattern in single_plus_three:
            if trainer.train_fidelity_losses[pattern]:
                loss_values = [max(eps, float(v)) for v in trainer.train_fidelity_losses[pattern]]
                linewidth = 2 if pattern == '3L_all' else 1
                axes[0, 1].plot(loss_values, label=pattern, color=colors_single_three[pattern], linewidth=linewidth)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (log scale)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, which='both', alpha=0.3)
        
        # axes[1, 0]: Two-level combinations only (train)
        two_patterns = ['2L_1+2', '2L_1+3', '2L_2+3']
        colors_two = {'2L_1+2': 'cyan', '2L_1+3': 'magenta', '2L_2+3': 'yellow'}
        axes[1, 0].set_title('Two-Level Combinations (Train)')
        for pattern in two_patterns:
            if trainer.train_fidelity_losses[pattern]:
                loss_values = [max(eps, float(v)) for v in trainer.train_fidelity_losses[pattern]]
                axes[1, 0].plot(loss_values, label=pattern, color=colors_two[pattern])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (log scale)')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, which='both', alpha=0.3)
        
        # axes[1, 1]: 3L_all comparison (train + val)
        axes[1, 1].set_title('3L_all: Train vs Val')
        if trainer.train_fidelity_losses['3L_all']:
            loss_values = [max(eps, float(v)) for v in trainer.train_fidelity_losses['3L_all']]
            axes[1, 1].plot(loss_values, label='Train 3L_all', color='blue', linewidth=2)
        if trainer.val_fidelity_losses['3L_all']:
            loss_values = [max(eps, float(v)) for v in trainer.val_fidelity_losses['3L_all']]
            axes[1, 1].plot(loss_values, label='Val 3L_all', 
                          color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (log scale)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to: {save_path}")
        
    def save_test_results(self, df_results):
        """Save test evaluation results"""
        print("\nSaving test results...")
        
        # Save as CSV
        csv_path = self.results_dir / "test_results.csv"
        df_results.to_csv(csv_path, index=False)
        
        # Save as formatted text
        txt_path = self.results_dir / "test_results.txt"
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TEST EVALUATION RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(df_results.to_string(index=False))
            f.write("\n\n")
            f.write("="*80 + "\n")
            
        print(f"✓ Test results saved:")
        print(f"  CSV: {csv_path}")
        print(f"  TXT: {txt_path}")
        
    def save_sample_predictions(self, vec_low, vec_med, vec_high, vec_true, 
                                y_dmd_pred, region_mask, mask_land, x, y, 
                                year_indices=[0], day_indices=[0, 92, 182, 274]):
        """Save sample prediction visualizations for multiple test years and seasons
        
        Args:
            vec_low, vec_med, vec_high, vec_true: Prediction tensors (already SIC values)
            y_dmd_pred: DMD baseline predictions
            region_mask: Active region mask
            mask_land: Land mask
            x, y: Spatial coordinates
            year_indices: List of test year indices to plot (default: [0] for first test year)
            day_indices: List of day indices to plot (default: [0,92,182,274] for 4 seasons)
        
        Note: If multiple years, plots different season per year (cycles through day_indices)
        """
        print(f"\nSaving sample predictions for {len(year_indices)} year(s), {len(day_indices)} day(s) each...")
        
        print(f"\nSaving sample predictions for {len(year_indices)} year(s), {len(day_indices)} day(s) each...")
        
        # Get spatial dimensions from DMD predictions
        ny, nx = y_dmd_pred.shape[2], y_dmd_pred.shape[3]
        print(f"  Spatial dimensions: {ny} x {nx}")
        
        # Check mask compatibility
        if mask_land.shape != (ny, nx):
            print(f"  ⚠️  Warning: mask_land shape {mask_land.shape} != expected {(ny, nx)}")
        
        # Create plots for each year-day combination
        for i, year_idx in enumerate(year_indices):
            # Cycle through day_indices if multiple years
            day_idx = day_indices[i % len(day_indices)]
            
            print(f"  Creating plot for Year {year_idx}, Day {day_idx}...")
            
            # Extract single day (these are already full SIC values, not residuals)
            true_seq = vec_true[0, day_idx].detach().cpu().numpy()
            pred_seq = vec_med[0, day_idx].detach().cpu().numpy()
            low_seq = vec_low[0, day_idx].detach().cpu().numpy()
            high_seq = vec_high[0, day_idx].detach().cpu().numpy()
            high_seq = vec_high[0, day_idx].detach().cpu().numpy()
            
            # Initialize full spatial grids with DMD baseline (for non-predicted regions)
            dmd_day = y_dmd_pred[year_idx, day_idx]
            pred_full = dmd_day.copy()
            low_full = dmd_day.copy()
            high_full = dmd_day.copy()
            true_full = dmd_day.copy()
            
            # Fill in predicted region with full SIC values (not residuals!)
            # These are already DMD + residual, clamped to [0,1]
            mask_flat = region_mask.reshape(-1)
            pred_full_flat = pred_full.reshape(-1)
            low_full_flat = low_full.reshape(-1)
            high_full_flat = high_full.reshape(-1)
            true_full_flat = true_full.reshape(-1)
            
            pred_full_flat[mask_flat] = pred_seq
            low_full_flat[mask_flat] = low_seq
            high_full_flat[mask_flat] = high_seq
            true_full_flat[mask_flat] = true_seq
            
            pred_full = pred_full_flat.reshape(ny, nx)
            low_full = low_full_flat.reshape(ny, nx)
            high_full = high_full_flat.reshape(ny, nx)
            true_full = true_full_flat.reshape(ny, nx)
            
            # Determine season name
            season_names = {0: 'Summer', 92: 'Autumn', 182: 'Winter', 274: 'Spring'}
            season = season_names.get(day_idx, f'Day{day_idx}')
            
            # Plot
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Test Year {year_idx}, {season} (Day {day_idx})', fontsize=16, fontweight='bold')
            
            extent = [x.min(), x.max(), y.min(), y.max()]
            
            # True
            im0 = axes[0, 0].imshow(true_full, origin='lower', cmap='viridis', 
                                    extent=extent, vmin=0, vmax=1)
            axes[0, 0].set_title(f'Ground Truth')
            plt.colorbar(im0, ax=axes[0, 0])
            
            # DMD Baseline
            im1 = axes[0, 1].imshow(dmd_day, origin='lower', cmap='viridis', 
                                    extent=extent, vmin=0, vmax=1)
            axes[0, 1].set_title('DMD Baseline')
            plt.colorbar(im1, ax=axes[0, 1])
            
            # Hybrid Prediction
            im2 = axes[0, 2].imshow(pred_full, origin='lower', cmap='viridis', 
                                    extent=extent, vmin=0, vmax=1)
            axes[0, 2].set_title('Hybrid Prediction (DMD + Model)')
            plt.colorbar(im2, ax=axes[0, 2])
            
            # Lower Bound
            im3 = axes[1, 0].imshow(low_full, origin='lower', cmap='viridis', 
                                    extent=extent, vmin=0, vmax=1)
            axes[1, 0].set_title('Lower Bound (90% CI)')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Upper Bound
            im4 = axes[1, 1].imshow(high_full, origin='lower', cmap='viridis', 
                                    extent=extent, vmin=0, vmax=1)
            axes[1, 1].set_title('Upper Bound (90% CI)')
            plt.colorbar(im4, ax=axes[1, 1])
            
            # Uncertainty (width)
            uncertainty = high_full - low_full
            im5 = axes[1, 2].imshow(uncertainty, origin='lower', cmap='hot', 
                                    extent=extent, vmin=0, vmax=0.5)
            axes[1, 2].set_title('Uncertainty (CI Width)')
            plt.colorbar(im5, ax=axes[1, 2])
            
            plt.tight_layout()
            save_path = self.results_dir / f"predictions_year{year_idx}_{season}_day{day_idx}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ Saved: {save_path.name}")
        
        print(f"\n✓ All sample predictions saved")
        
    def save_all_data(self, q_scores, scenario_predictions):
        """Save all numerical data for ALL scenarios for post-processing (with final [0,1] clipping)
        
        Args:
            q_scores: Dict[scenario_name] -> conformity scores array
            scenario_predictions: Dict[scenario_name] -> (vec_low, vec_med, vec_high, vec_true)
        """
        print("\nSaving all numerical data for ALL scenarios...")
        
        # Build predictions dict with all scenarios
        predictions_dict = {}
        
        for scenario_name, (vec_low, vec_med, vec_high, vec_true) in scenario_predictions.items():
            # Convert to numpy and ensure [0,1] clipping
            predictions_dict[scenario_name] = {
                'low': torch.clamp(vec_low, 0, 1).cpu().numpy(),
                'median': torch.clamp(vec_med, 0, 1).cpu().numpy(),
                'high': torch.clamp(vec_high, 0, 1).cpu().numpy(),
                'true': torch.clamp(vec_true, 0, 1).cpu().numpy()
            }
            print(f"  ✓ {scenario_name}: shapes {predictions_dict[scenario_name]['median'].shape}")
        
        # Create final data structure
        data_dict = {
            'q_scores': q_scores,
            'predictions': predictions_dict
        }
        
        save_path = self.results_dir / "all_predictions.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
            
        print(f"\n✓ All numerical data saved to: {save_path}")
        print(f"  Scenarios saved: {list(predictions_dict.keys())}")
    
    def save_icenet_metrics(self, scenario_predictions, region_mask, mask_land, x, y, grid_cell_area_km2=625.0):
        """
        Compute and save IceNet-style metrics for comparison.
        
        This includes:
        1. IIEE and Binary Accuracy (overall and seasonal)
        2. Mean Prediction Interval Width in MIZ
        3. Calibration/Reliability diagrams
        4. CI width spatial maps
        
        Args:
            scenario_predictions: Dict[scenario_name] -> (vec_low, vec_med, vec_high, vec_true)
            region_mask: (ny, nx) Boolean mask for active region
            mask_land: (ny, nx) Boolean mask for land
            x, y: Coordinate arrays
            grid_cell_area_km2: Area of each grid cell in km² (default: 625 for 25km grid)
        """
        print("\n" + "="*80)
        print("COMPUTING ICENET-STYLE METRICS")
        print("="*80)
        
        # Initialize IceNet metrics calculator
        icenet = IceNetMetrics(grid_cell_area_km2=grid_cell_area_km2)
        
        # Storage for all metrics
        all_metrics = {}
        
        # Process each scenario
        for scenario_name, (vec_low, vec_med, vec_high, vec_true) in scenario_predictions.items():
            print(f"\n  Processing scenario: {scenario_name}")
            
            # Convert to numpy if needed
            if torch.is_tensor(vec_med):
                vec_low_np = vec_low.cpu().numpy()
                vec_med_np = vec_med.cpu().numpy()
                vec_high_np = vec_high.cpu().numpy()
                vec_true_np = vec_true.cpu().numpy()
            else:
                vec_low_np = vec_low
                vec_med_np = vec_med
                vec_high_np = vec_high
                vec_true_np = vec_true
            
            # Reshape to (T, n_pixels) - assuming shape is (B, S, n_pixels) or (S, n_pixels)
            if vec_med_np.ndim == 3:  # (B, S, n_pixels)
                B, S, n_pixels = vec_med_np.shape
                vec_low_np = vec_low_np.reshape(-1, n_pixels)
                vec_med_np = vec_med_np.reshape(-1, n_pixels)
                vec_high_np = vec_high_np.reshape(-1, n_pixels)
                vec_true_np = vec_true_np.reshape(-1, n_pixels)
            
            scenario_metrics = {}
            
            # 1. OVERALL IIEE AND BINARY ACCURACY
            print("    Computing IIEE and Binary Accuracy...")
            binary_acc, iiee, total_area, O, U = icenet.calculate_binary_accuracy(
                vec_med_np, vec_true_np, active_mask=None, ice_threshold=0.15
            )
            
            scenario_metrics['overall'] = {
                'binary_accuracy_%': binary_acc,
                'iiee_km2': iiee,
                'overestimation_km2': O,
                'underestimation_km2': U,
                'total_area_km2': total_area
            }
            
            print(f"      Binary Accuracy: {binary_acc:.2f}%")
            print(f"      IIEE: {iiee:.0f} km²")
            print(f"        Overestimation:  {O:.0f} km²")
            print(f"        Underestimation: {U:.0f} km²")
            
            # 2. SEASONAL IIEE AND BINARY ACCURACY
            print("    Computing seasonal metrics...")
            seasonal_metrics = icenet.compute_seasonal_iiee_binary_accuracy(
                vec_med_np, vec_true_np, days_per_year=365, n_seasons=4
            )
            scenario_metrics['seasonal'] = seasonal_metrics
            
            for season, metrics in seasonal_metrics.items():
                print(f"      {season:10s}: Binary Acc = {metrics['binary_accuracy']:.2f}%, "
                      f"IIEE = {metrics['iiee']:.0f} km²")
            
            # 3. MIZ ANALYSIS
            print("    Computing MIZ CI width metrics...")
            mpiw_miz, mpiw_all, miz_fraction = icenet.calculate_miz_ci_width(
                vec_low_np, vec_high_np, vec_true_np, miz_lower=0.15, miz_upper=0.80
            )
            
            scenario_metrics['miz'] = {
                'mpiw_in_miz': mpiw_miz,
                'mpiw_overall': mpiw_all,
                'miz_fraction': miz_fraction
            }
            
            print(f"      MPIW in MIZ: {mpiw_miz:.4f}")
            print(f"      MPIW overall: {mpiw_all:.4f}")
            print(f"      Fraction of pixels in MIZ: {miz_fraction:.2%}")
            
            # 4. CALIBRATION/RELIABILITY DIAGRAM DATA
            print("    Computing calibration metrics...")
            coverage_overall, bin_centers, bin_coverage, bin_counts = \
                icenet.calculate_reliability_diagram_data(
                    vec_low_np, vec_high_np, vec_true_np, n_bins=10
                )
            
            scenario_metrics['calibration'] = {
                'coverage_overall': coverage_overall,
                'bin_centers': bin_centers,
                'bin_coverage': bin_coverage,
                'bin_counts': bin_counts,
                'target_coverage': 0.90
            }
            
            print(f"      Overall Coverage: {coverage_overall:.1%} (target: 90%)")
            
            # 5. SAVE RELIABILITY DIAGRAM
            reliability_path = self.results_dir / f"reliability_diagram_{scenario_name}.png"
            icenet.plot_reliability_diagram(
                coverage_overall, bin_centers, bin_coverage, bin_counts,
                target_coverage=0.90, save_path=reliability_path
            )
            
            # 6. COMPUTE AND SAVE CI WIDTH MAPS (SEASONAL)
            print("    Generating CI width maps for each season...")
            ny, nx = mask_land.shape
            n_timesteps = vec_med_np.shape[0]
            days_per_year = 365
            n_seasons = 4
            days_per_season = days_per_year // n_seasons
            season_names = ['Summer', 'Autumn', 'Winter', 'Spring']
            
            for season_idx in range(n_seasons):
                # Get seasonal indices
                season_mask = np.zeros(n_timesteps, dtype=bool)
                for year_start in range(0, n_timesteps, days_per_year):
                    season_start = year_start + season_idx * days_per_season
                    season_end = min(season_start + days_per_season, n_timesteps)
                    if season_start < n_timesteps:
                        season_mask[season_start:season_end] = True
                
                if np.sum(season_mask) > 0:
                    # Average CI width over season
                    widths_season = vec_high_np[season_mask] - vec_low_np[season_mask]
                    avg_width_season = np.mean(widths_season, axis=0)  # (n_pixels,)
                    
                    # Reshape to spatial grid
                    width_map = np.full((ny, nx), np.nan)
                    width_map[region_mask] = avg_width_season
                    
                    # Save map
                    ci_width_path = self.results_dir / f"ci_width_map_{scenario_name}_{season_names[season_idx]}.png"
                    icenet.plot_ci_width_map(
                        width_map, mask_land, x, y, 
                        season_name=f"{scenario_name} - {season_names[season_idx]}",
                        save_path=ci_width_path
                    )
            
            all_metrics[scenario_name] = scenario_metrics
        
        # Save all metrics to CSV and JSON
        print("\n  Saving metrics summary...")
        
        # Create summary DataFrame
        summary_data = []
        for scenario, metrics in all_metrics.items():
            row = {
                'Scenario': scenario,
                'Binary_Accuracy_%': metrics['overall']['binary_accuracy_%'],
                'IIEE_km2': metrics['overall']['iiee_km2'],
                'Overestimation_km2': metrics['overall']['overestimation_km2'],
                'Underestimation_km2': metrics['overall']['underestimation_km2'],
                'MPIW_in_MIZ': metrics['miz']['mpiw_in_miz'],
                'MPIW_overall': metrics['miz']['mpiw_overall'],
                'MIZ_Fraction': metrics['miz']['miz_fraction'],
                'Coverage': metrics['calibration']['coverage_overall']
            }
            
            # Add seasonal binary accuracy
            for season, season_metrics in metrics['seasonal'].items():
                row[f'Binary_Accuracy_{season}_%'] = season_metrics['binary_accuracy']
                row[f'IIEE_{season}_km2'] = season_metrics['iiee']
            
            summary_data.append(row)
        
        df_icenet = pd.DataFrame(summary_data)
        
        # Save CSV
        csv_path = self.results_dir / "icenet_metrics.csv"
        df_icenet.to_csv(csv_path, index=False)
        print(f"    ✓ Saved CSV to: {csv_path}")
        
        # Save formatted text
        txt_path = self.results_dir / "icenet_metrics.txt"
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ICENET-STYLE METRICS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(df_icenet.to_string(index=False))
            f.write("\n\n")
            
            # Add detailed seasonal breakdown
            f.write("="*80 + "\n")
            f.write("DETAILED SEASONAL BREAKDOWN\n")
            f.write("="*80 + "\n\n")
            
            for scenario, metrics in all_metrics.items():
                f.write(f"\nScenario: {scenario}\n")
                f.write("-"*40 + "\n")
                for season, season_metrics in metrics['seasonal'].items():
                    f.write(f"  {season:10s}:\n")
                    f.write(f"    Binary Accuracy: {season_metrics['binary_accuracy']:6.2f}%\n")
                    f.write(f"    IIEE:            {season_metrics['iiee']:10.0f} km²\n")
                    f.write(f"    Overestimation:  {season_metrics['overestimation']:10.0f} km²\n")
                    f.write(f"    Underestimation: {season_metrics['underestimation']:10.0f} km²\n")
        
        print(f"    ✓ Saved formatted text to: {txt_path}")
        
        # Save full detailed metrics as pickle
        pkl_path = self.results_dir / "icenet_metrics_full.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(all_metrics, f)
        print(f"    ✓ Saved full metrics to: {pkl_path}")
        
        print("\n" + "="*80)
        print("✅ ICENET METRICS COMPUTATION COMPLETE")
        print("="*80)
        print("\nSummary Table:")
        print(df_icenet.to_string(index=False))
        
        return all_metrics, df_icenet
    
    def save_scenario_day_predictions(self, scenario_predictions, y_dmd_pred, region_mask, mask_land, x, y, 
                                       n_year_train, n_year_val, n_year_test, target_day=180):
        """Save day 180 predictions for ALL test years and ALL scenarios
        
        Args:
            scenario_predictions: Dict[scenario_name] -> (vec_low, vec_med, vec_high, vec_true) tensors
            y_dmd_pred: Full DMD predictions array
            region_mask: Active region mask
            mask_land: Land mask
            x, y: Coordinate arrays
            n_year_train, n_year_val, n_year_test: Split sizes
            target_day: Day of year to visualize (default 180)
        """
        print(f"\n📊 SAVING DAY {target_day} PREDICTIONS FOR ALL SCENARIOS AND TEST YEARS")
        print("-"*70)
        
        ny, nx = y_dmd_pred.shape[2], y_dmd_pred.shape[3]
        extent = [x.min(), x.max(), y.min(), y.max()]
        
        # Create subdirectory for scenario comparisons
        scenario_dir = self.results_dir / "scenario_predictions"
        scenario_dir.mkdir(exist_ok=True)
        
        # Iterate over test years
        for test_year_idx in range(n_year_test):
            absolute_year_idx = n_year_train + n_year_val + test_year_idx
            
            print(f"\n  Processing Test Year {test_year_idx + 1}/{n_year_test} (Absolute Year {absolute_year_idx})")
            
            # --- GENERATE COMBINED PLOT FOR ALL SCENARIOS ---
            n_scenarios = len(scenario_predictions)
            fig, axes = plt.subplots(n_scenarios, 5, figsize=(25, 5 * n_scenarios))
            if n_scenarios == 1:
                axes = axes.reshape(1, -1)
            
            for scenario_idx, (scenario_name, (vec_low, vec_med, vec_high, vec_true)) in enumerate(scenario_predictions.items()):
                # Extract day 180 for this test year
                day_idx_in_test = test_year_idx * 365 + target_day
                
                true_seq = vec_true[0, day_idx_in_test].detach().cpu().numpy()
                pred_seq = vec_med[0, day_idx_in_test].detach().cpu().numpy()
                low_seq = vec_low[0, day_idx_in_test].detach().cpu().numpy()
                high_seq = vec_high[0, day_idx_in_test].detach().cpu().numpy()
                
                # Get DMD baseline for this day
                dmd_day = y_dmd_pred[absolute_year_idx, target_day]
                
                # Reconstruct full spatial grids
                pred_full = dmd_day.copy()
                low_full = dmd_day.copy()
                high_full = dmd_day.copy()
                true_full = dmd_day.copy()
                
                mask_flat = region_mask.reshape(-1)
                pred_full.reshape(-1)[mask_flat] = pred_seq
                low_full.reshape(-1)[mask_flat] = low_seq
                high_full.reshape(-1)[mask_flat] = high_seq
                true_full.reshape(-1)[mask_flat] = true_seq
                
                pred_full = pred_full.reshape(ny, nx)
                low_full = low_full.reshape(ny, nx)
                high_full = high_full.reshape(ny, nx)
                true_full = true_full.reshape(ny, nx)
                
                # Plot row for this scenario
                ax = axes[scenario_idx]
                
                # Ground Truth
                im0 = ax[0].imshow(true_full, origin='lower', cmap='viridis', 
                                   extent=extent, vmin=0, vmax=1)
                ax[0].set_title(f'{scenario_name}\nGround Truth')
                plt.colorbar(im0, ax=ax[0], fraction=0.046)
                
                # DMD Baseline
                im1 = ax[1].imshow(dmd_day, origin='lower', cmap='viridis', 
                                   extent=extent, vmin=0, vmax=1)
                ax[1].set_title('DMD Baseline')
                plt.colorbar(im1, ax=ax[1], fraction=0.046)
                
                # Model Prediction
                im2 = ax[2].imshow(pred_full, origin='lower', cmap='viridis', 
                                   extent=extent, vmin=0, vmax=1)
                ax[2].set_title('Model Prediction')
                plt.colorbar(im2, ax=ax[2], fraction=0.046)
                
                # Uncertainty (CI Width)
                uncertainty = high_full - low_full
                im3 = ax[3].imshow(uncertainty, origin='lower', cmap='hot', 
                                   extent=extent, vmin=0, vmax=0.5)
                ax[3].set_title('Uncertainty (CI Width)')
                plt.colorbar(im3, ax=ax[3], fraction=0.046)
                
                # Model Error
                error = np.abs(pred_full - true_full)
                im4 = ax[4].imshow(error, origin='lower', cmap='Reds', 
                                   extent=extent, vmin=0, vmax=0.3)
                ax[4].set_title('Absolute Error')
                plt.colorbar(im4, ax=ax[4], fraction=0.046)
            
            plt.suptitle(f'Day {target_day} Predictions - Test Year {test_year_idx + 1} (All Scenarios)', 
                        fontsize=16, y=0.995)
            plt.tight_layout()
            
            save_path = scenario_dir / f"day{target_day}_testyear{test_year_idx + 1}_all_scenarios.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ Saved: {save_path.name}")
        
        print("\n✓ All scenario predictions saved")
    
    def save_seasonal_analysis(self, scenario_predictions, y_dmd_pred, climatology_avg, 
                              region_mask, mask_land, x, y, n_year_train, n_year_val, n_year_test):
        """Save comprehensive seasonal analysis: spatial maps, time series, and box plots
        
        Args:
            scenario_predictions: Dict[scenario_name] -> (vec_low, vec_med, vec_high, vec_true)
            y_dmd_pred: DMD baseline predictions
            climatology_avg: Climatological average (365, ny, nx)
            region_mask: Active region mask
            mask_land: Land mask
            x, y: Coordinate arrays
            n_year_train, n_year_val, n_year_test: Split sizes
        """
        print("\n📊 GENERATING SEASONAL ANALYSIS")
        print("-"*70)
        
        ny, nx = y_dmd_pred.shape[2], y_dmd_pred.shape[3]
        extent = [x.min(), x.max(), y.min(), y.max()]
        
        # Create subdirectory
        seasonal_dir = self.results_dir / "seasonal_analysis"
        seasonal_dir.mkdir(exist_ok=True)
        
        # Season definitions (Antarctic)
        season_names = ['Summer', 'Autumn', 'Winter', 'Spring']
        season_days = {
            'Summer': list(range(335, 365)) + list(range(0, 60)),
            'Autumn': list(range(60, 152)),
            'Winter': list(range(152, 244)),
            'Spring': list(range(244, 335))
        }
        
        # --- PART 1: SPATIAL MAPS OF UNCERTAINTY PER SEASON ---
        print("\n  1. Generating seasonal spatial uncertainty maps...")
        for scenario_name, (vec_low, vec_med, vec_high, vec_true) in scenario_predictions.items():
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            
            for season_idx, (season_name, days) in enumerate(season_days.items()):
                # Average uncertainty over season and test years
                seasonal_widths = []
                seasonal_coverage = []
                
                for test_year in range(n_year_test):
                    for day in days:
                        day_idx = test_year * 365 + day
                        if day_idx < vec_low.shape[1]:  # Safety check
                            low = vec_low[0, day_idx].detach().cpu().numpy()
                            high = vec_high[0, day_idx].detach().cpu().numpy()
                            true = vec_true[0, day_idx].detach().cpu().numpy()
                            
                            width = high - low
                            coverage = ((true >= low) & (true <= high)).astype(float)
                            
                            seasonal_widths.append(width)
                            seasonal_coverage.append(coverage)
                
                # Average over all days in season
                avg_width = np.mean(seasonal_widths, axis=0)
                avg_coverage = np.mean(seasonal_coverage, axis=0)
                
                # Reconstruct spatial maps
                width_map = np.full((ny, nx), np.nan)
                coverage_map = np.full((ny, nx), np.nan)
                
                mask_flat = region_mask.reshape(-1)
                width_map.reshape(-1)[mask_flat] = avg_width
                coverage_map.reshape(-1)[mask_flat] = avg_coverage
                
                # Plot uncertainty width
                ax = axes[0, season_idx]
                im = ax.imshow(width_map, origin='lower', cmap='hot', 
                               extent=extent, vmin=0, vmax=0.4)
                ax.set_title(f'{season_name}\nAvg CI Width')
                plt.colorbar(im, ax=ax, fraction=0.046)
                
                # Plot coverage
                ax = axes[1, season_idx]
                im = ax.imshow(coverage_map, origin='lower', cmap='RdYlGn', 
                               extent=extent, vmin=0.8, vmax=1.0)
                ax.set_title(f'{season_name}\nCoverage Rate')
                plt.colorbar(im, ax=ax, fraction=0.046)
            
            plt.suptitle(f'Seasonal Uncertainty Analysis - {scenario_name}', fontsize=16)
            plt.tight_layout()
            
            save_path = seasonal_dir / f"spatial_maps_{scenario_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ Saved spatial maps: {scenario_name}")
        
        # --- PART 2: TIME SERIES OF COVERAGE AND WIDTH PER SEASON ---
        print("\n  2. Generating seasonal time series...")
        for scenario_name, (vec_low, vec_med, vec_high, vec_true) in scenario_predictions.items():
            fig, axes = plt.subplots(4, 2, figsize=(16, 20))
            
            for season_idx, (season_name, days) in enumerate(season_days.items()):
                # Collect metrics for each test year
                yearly_width = []
                yearly_coverage = []
                
                for test_year in range(n_year_test):
                    day_widths = []
                    day_coverage = []
                    
                    for day in days:
                        day_idx = test_year * 365 + day
                        if day_idx < vec_low.shape[1]:
                            low = vec_low[0, day_idx].detach().cpu().numpy()
                            high = vec_high[0, day_idx].detach().cpu().numpy()
                            true = vec_true[0, day_idx].detach().cpu().numpy()
                            
                            width = high - low
                            coverage = ((true >= low) & (true <= high)).astype(float)
                            
                            day_widths.append(width.mean())
                            day_coverage.append(coverage.mean())
                    
                    yearly_width.append(day_widths)
                    yearly_coverage.append(day_coverage)
                
                # Plot width time series
                ax = axes[season_idx, 0]
                
                # Handle summer season crossing year boundary
                if season_name == 'Summer':
                    # Create continuous x-axis: days 335-364 stay as is, days 0-59 become 365-424
                    x_continuous = [d if d >= 335 else d + 365 for d in days]
                    for year_idx, widths in enumerate(yearly_width):
                        ax.plot(x_continuous[:len(widths)], widths, label=f'Year {year_idx + 1}', alpha=0.7)
                    # Add vertical line at year transition (day 365)
                    ax.axvline(x=365, color='gray', linestyle=':', alpha=0.5)
                    ax.text(365, ax.get_ylim()[1] * 0.95, 'Year t+1', ha='left', va='top', fontsize=8, color='gray')
                    # Update x-ticks to show actual day of year
                    xticks = ax.get_xticks()
                    xticklabels = [int(x) if x < 365 else int(x - 365) for x in xticks]
                    ax.set_xticklabels(xticklabels)
                else:
                    for year_idx, widths in enumerate(yearly_width):
                        ax.plot(days[:len(widths)], widths, label=f'Year {year_idx + 1}', alpha=0.7)
                
                ax.set_title(f'{season_name} - CI Width Over Time')
                ax.set_xlabel('Day of Year')
                ax.set_ylabel('Average CI Width')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Target')
                
                # Plot coverage time series
                ax = axes[season_idx, 1]
                
                # Handle summer season crossing year boundary
                if season_name == 'Summer':
                    # Create continuous x-axis: days 335-364 stay as is, days 0-59 become 365-424
                    x_continuous = [d if d >= 335 else d + 365 for d in days]
                    for year_idx, coverage in enumerate(yearly_coverage):
                        ax.plot(x_continuous[:len(coverage)], coverage, label=f'Year {year_idx + 1}', alpha=0.7)
                    # Add vertical line at year transition (day 365)
                    ax.axvline(x=365, color='gray', linestyle=':', alpha=0.5)
                    ax.text(365, 0.96, 'Year t+1', ha='left', va='top', fontsize=8, color='gray')
                    # Update x-ticks to show actual day of year
                    xticks = ax.get_xticks()
                    xticklabels = [int(x) if x < 365 else int(x - 365) for x in xticks]
                    ax.set_xticklabels(xticklabels)
                else:
                    for year_idx, coverage in enumerate(yearly_coverage):
                        ax.plot(days[:len(coverage)], coverage, label=f'Year {year_idx + 1}', alpha=0.7)
                
                ax.set_title(f'{season_name} - Coverage Over Time')
                ax.set_xlabel('Day of Year')
                ax.set_ylabel('Coverage Rate')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Target 90%')
                ax.set_ylim([0.7, 1.0])
            
            plt.suptitle(f'Seasonal Time Series - {scenario_name}', fontsize=16)
            plt.tight_layout()
            
            save_path = seasonal_dir / f"timeseries_{scenario_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ Saved time series: {scenario_name}")
        
        # --- PART 3: BOX PLOTS OF CI WIDTH DISTRIBUTION PER SEASON ---
        print("\n  3. Generating seasonal box plots...")
        for scenario_name, (vec_low, vec_med, vec_high, vec_true) in scenario_predictions.items():
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Collect data for all seasons
            season_width_data = []
            season_coverage_data = []
            
            for season_name, days in season_days.items():
                width_values = []
                coverage_values = []
                
                for test_year in range(n_year_test):
                    for day in days:
                        day_idx = test_year * 365 + day
                        if day_idx < vec_low.shape[1]:
                            low = vec_low[0, day_idx].detach().cpu().numpy()
                            high = vec_high[0, day_idx].detach().cpu().numpy()
                            true = vec_true[0, day_idx].detach().cpu().numpy()
                            
                            width = high - low
                            coverage = ((true >= low) & (true <= high)).astype(float)
                            
                            width_values.extend(width.flatten())
                            coverage_values.extend(coverage.flatten())
                
                season_width_data.append(width_values)
                season_coverage_data.append(coverage_values)
            
            # Box plot for CI width
            ax = axes[0]
            bp = ax.boxplot(season_width_data, labels=season_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.set_title('CI Width Distribution by Season')
            ax.set_ylabel('CI Width')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Box plot for coverage
            ax = axes[1]
            bp = ax.boxplot(season_coverage_data, labels=season_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightgreen')
            ax.set_title('Coverage Rate Distribution by Season')
            ax.set_ylabel('Coverage Rate')
            ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target 90%')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
            
            plt.suptitle(f'Seasonal Distribution Analysis - {scenario_name}', fontsize=16)
            plt.tight_layout()
            
            save_path = seasonal_dir / f"boxplots_{scenario_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ Saved box plots: {scenario_name}")
        
        print("\n✓ Seasonal analysis complete")
    
    def save_baseline_comparisons(self, scenario_predictions, dmd_baseline_active, climatology_baseline_active):
        """Compare model predictions against DMD and climatology baselines
        
        Args:
            scenario_predictions: Dict[scenario_name] -> (vec_low, vec_med, vec_high, vec_true)
            dmd_baseline_active: Torch tensor (n_timesteps, n_pixels) - DMD on active region
            climatology_baseline_active: Torch tensor (n_timesteps, n_pixels) - Climatology on active region
        """
        print("\n📊 GENERATING BASELINE COMPARISONS")
        print("-"*70)
        print(f"  DMD baseline shape: {dmd_baseline_active.shape}")
        print(f"  Climatology baseline shape: {climatology_baseline_active.shape}")
        
        comparison_dir = self.results_dir / "baseline_comparisons"
        comparison_dir.mkdir(exist_ok=True)
        
        # Compute metrics for each scenario
        results_list = []
        
        for scenario_name, (vec_low, vec_med, vec_high, vec_true) in scenario_predictions.items():
            print(f"  Processing {scenario_name}...")
            
            # Verify shapes match
            n_timesteps = vec_true.shape[1]
            print(f"    Predictions shape: {vec_med.shape}, True shape: {vec_true.shape}")
            print(f"    DMD available: {dmd_baseline_active.shape[0]}, Needed: {n_timesteps}")
            
            # Use pre-computed baselines (already on active region)
            dmd_active = dmd_baseline_active[:n_timesteps]
            clim_active = climatology_baseline_active[:n_timesteps]
            
            # Compute MAE for each baseline and model
            mae_model = torch.abs(vec_med[0] - vec_true[0]).mean().item()
            mae_dmd = torch.abs(dmd_active - vec_true[0]).mean().item()
            mae_clim = torch.abs(clim_active - vec_true[0]).mean().item()
            
            print(f"    MAE - Model: {mae_model:.6f}, DMD: {mae_dmd:.6f}, Clim: {mae_clim:.6f}")
            
            # Compute improvement
            improvement_vs_dmd = (mae_dmd - mae_model) / mae_dmd * 100
            improvement_vs_clim = (mae_clim - mae_model) / mae_clim * 100
            
            # Compute coverage and width
            coverage = ((vec_true[0] >= vec_low[0]) & (vec_true[0] <= vec_high[0])).float().mean().item()
            width = (vec_high[0] - vec_low[0]).mean().item()
            
            results_list.append({
                'Scenario': scenario_name,
                'MAE_Model': mae_model,
                'MAE_DMD': mae_dmd,
                'MAE_Climatology': mae_clim,
                'Improvement_vs_DMD_%': improvement_vs_dmd,
                'Improvement_vs_Clim_%': improvement_vs_clim,
                'Coverage': coverage,
                'CI_Width': width
            })
        
        df = pd.DataFrame(results_list)
        
        # Save as CSV
        csv_path = comparison_dir / "baseline_comparison_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  ✓ Saved metrics: {csv_path.name}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        scenarios = df['Scenario'].values
        x_pos = np.arange(len(scenarios))
        width = 0.25
        
        # Plot 1: MAE Comparison
        ax = axes[0, 0]
        ax.bar(x_pos - width, df['MAE_Model'], width, label='Model', color='blue', alpha=0.8)
        ax.bar(x_pos, df['MAE_DMD'], width, label='DMD', color='orange', alpha=0.8)
        ax.bar(x_pos + width, df['MAE_Climatology'], width, label='Climatology', color='green', alpha=0.8)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('MAE Comparison Across Baselines')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Improvement Percentages
        ax = axes[0, 1]
        ax.bar(x_pos - width/2, df['Improvement_vs_DMD_%'], width, label='vs DMD', color='blue', alpha=0.8)
        ax.bar(x_pos + width/2, df['Improvement_vs_Clim_%'], width, label='vs Climatology', color='green', alpha=0.8)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Relative Improvement Over Baselines')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Coverage vs Width
        ax = axes[1, 0]
        scatter = ax.scatter(df['CI_Width'], df['Coverage'], 
                            c=range(len(df)), cmap='viridis', s=200, alpha=0.7)
        for idx, scenario in enumerate(scenarios):
            ax.annotate(scenario, (df['CI_Width'].iloc[idx], df['Coverage'].iloc[idx]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target Coverage')
        ax.set_xlabel('Average CI Width')
        ax.set_ylabel('Coverage Rate')
        ax.set_title('Coverage vs Uncertainty Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Ranked Performance
        ax = axes[1, 1]
        df_sorted = df.sort_values('MAE_Model')
        colors_ranked = ['green' if x > 0 else 'red' for x in df_sorted['Improvement_vs_DMD_%']]
        ax.barh(range(len(df_sorted)), df_sorted['Improvement_vs_DMD_%'], color=colors_ranked, alpha=0.7)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['Scenario'])
        ax.set_xlabel('Improvement vs DMD (%)')
        ax.set_title('Scenario Ranking by Performance')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = comparison_dir / "baseline_comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plots: {plot_path.name}")
        print("\n✓ Baseline comparison complete")
        print("\nSummary Table:")
        print(df.to_string(index=False))


# ============================================================================
# PART 7 COMPLETE
# ============================================================================


def main():
    """Main execution pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multifidelity Transformer GPU Execution")
    
    # --- MODIFICA CLUSTER ---
    parser.add_argument("--base_path", type=str, 
                       default="/work/u10715220",  # Default su WORK per i dati
                       help="Path to DATA and RESULTS (WORK)")
                       
    parser.add_argument("--project_path", type=str, 
                       default=".",  # Default alla cartella corrente (HOME) per il codice
                       help="Path to code and configuration files")
    # ------------------------

    parser.add_argument("--scratch_path", type=str, 
                       default="/scratch_global/u10715220", 
                       help="Path to SCRATCH directory for heavy files")
    parser.add_argument("--experiment_name", type=str, default="ice",
                       help="Experiment name")
    parser.add_argument("--experiment_number", type=int, default=21,
                       help="Experiment number")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="Device to use for computation")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training (load existing model)")
    # ... dopo --scratch_path ...
    parser.add_argument("--output_dir", type=str, default=None,
                    help="Path for OUTPUTS (Results, Checkpoints). Defaults to base_path if not set.")
        
    args = parser.parse_args()
    
    print("="*80)
    print("MULTIFIDELITY TRANSFORMER - FULL PIPELINE EXECUTION")
    print("="*80)
    print(f"Experiment: {args.experiment_name}_{args.experiment_number}")
    print(f"Device: {args.device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # ========================================================================
    # PHASE 1: CONFIGURATION AND DATA LOADING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: CONFIGURATION AND DATA LOADING")
    print("="*80)
    
    config = Config(args)
    
    # Load and preprocess data
    data_loader = DataLoader_Ice(config)
    mask_land_, mask_ice_, data_, data_mean_month_, data_mean_week_, x_, y_, thickness_data_, sst_data_ = data_loader.load_raw_data()
    data, data_mean_month, data_mean_week, x, y, mask_ice, mask_land, ny, nx, thickness_data, sst_data, bbox = \
        data_loader.preprocess_data(mask_land_, mask_ice_, data_, data_mean_month_, data_mean_week_, x_, y_, thickness_data_, sst_data_)
    
    # Split data (selecting years 1993-2023 by slicing [4:-1])
    data = data[4:-1]
    thickness_data = thickness_data[:len(data)]  # Match data length
    sst_data = sst_data[:len(data)]  # Match data length
    
    data_train, data_val, data_test, data_tot, n_year_train, n_year_val, n_year_test, ny, nx = \
        data_loader.split_train_val_test(data)
    
    n_year_tot = n_year_train + n_year_val + n_year_test
    
    # ========================================================================
    # PHASE 2: BASELINE COMPUTATIONS
    # ========================================================================

    # =========================================================================
    # PHASE 3: LOAD PRE-COMPUTED DMD FORECASTS
    # =========================================================================
    from pathlib import Path
    
    # Set split configuration (using DESIRED_TEST = 4 from config)
    n_years_total = 31  # Years available in data
    
    n_year_train = 23
    n_year_val = 4
    n_year_test = 4
    
    # Verification
    current_total = n_year_train + n_year_val + n_year_test
    if current_total != n_years_total:
        print(f"⚠️  WARNING: Split sum ({current_total}) != Available Years ({n_years_total})")
        
    print(f"\n🔧 Split Configuration:")
    print(f"   Train: {n_year_train} years")
    print(f"   Val:   {n_year_val} years")
    print(f"   Test:  {n_year_test} years")
    
    # =========================================================================
    # LOAD PRE-COMPUTED DMD FORECASTS (WITH SPATIAL CROPPING)
    # =========================================================================
    print("\n🔍 LOADING PRE-COMPUTED DMD FORECASTS...")
    
    dmd_baseline = DMDBaseline(config, bbox=bbox)
    dmd_years, y_dmd_pred, y_dmd_std = dmd_baseline.load_dmd_predictions()
    
    if y_dmd_pred is None:
        raise RuntimeError("❌ CRITICAL: DMD predictions could not be loaded. Cannot proceed.")
    
    print(f"✓ DMD forecasts loaded and cropped to match data domain")
    print(f"  Shape: {y_dmd_pred.shape}")
    print(f"  Years: {dmd_years[0]} to {dmd_years[-1]} ({len(dmd_years)} years)")

        # ========================================================================
    # PHASE 3: LOW-FIDELITY DATA PREPARATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: LOW-FIDELITY DATA PREPARATION")
    print("="*80)
    
    # Define target region using probabilistic ice mask (>0 threshold)
    # NOTE: Two different probabilistic masks are used:
    #   1. Region mask (>0 threshold) - ONLY for target variable (y) during loss computation (ALL dynamic ice pixels)
    #   2. Sensor mask (50% threshold) - for sensor placement to ensure non-zero timeseries
    # 
    # IMPORTANT: x1 (thickness) and x2 (SST) use the FULL grid for POD decomposition,
    #            NOT the region_mask. This maximizes information retention before dimension reduction.
    # 
    # CLIPPING: All final predictions (baseline + residuals) are clipped to [0,1] to maintain probability constraints
    print("\nUSING ICE MASK FOR TARGET REGION")
    print("-"*70)
    
    # Use full ice mask (includes all pixels in static ice mask, even if sometimes zero)
    region_mask = mask_ice
    
    n_pixel_region = region_mask.sum()
    print(f"✓ Using ice mask as region mask")
    print(f"  Pixels in region: {n_pixel_region}")
    print(f"  Total grid pixels: {ny*nx}")
    print(f"  Region coverage of grid: {n_pixel_region/(ny*nx)*100:.1f}%")
    
    # Validation check
    if region_mask.sum() < (ny*nx) * 0.2:  # Less than 20% of grid
        print(f"  ⚠️  WARNING: Region mask covers only {n_pixel_region/(ny*nx)*100:.1f}% of grid.")
        print(f"             This might be correct if data is pre-masked to ice-prone regions only.")
        print(f"             Verify this is intentional in the input data files.")
    
    # Initialize low-fidelity data prep with new data sources
    lf_prep = LowFidelityDataPrep(config, data_tot, thickness_data, sst_data,
                                  x, y, mask_ice, mask_land, region_mask)
    
    # Level 1: Ice thickness
    x1_train, x1_val, x1_test = lf_prep.prepare_thickness_data(
        n_year_train, n_year_val, n_year_test, n_year_tot
    )
    
    # Level 2: SST
    x2_train, x2_val, x2_test = lf_prep.prepare_sst_data(
        n_year_train, n_year_val, n_year_test, n_year_tot
    )
    
    # Apply POD reduction with variance threshold or fixed modes
    # threshold can be:
    #   - int (e.g., 64): use exactly that many modes
    #   - float (e.g., 0.95): use modes that explain that variance percentage
    x1_train, x1_val, x1_test, x2_train, x2_val, x2_test, pod_data = \
        lf_prep.apply_pod_reduction(
            x1_train, x1_val, x1_test,
            x2_train, x2_val, x2_test,
            threshold=0.9  # Use modes that explain 90% of variance
        )
    
    # Save POD data for later reconstruction if needed
    pod_path = config.checkpoint_dir / 'pod_data.pkl'
    with open(pod_path, 'wb') as f:
        pickle.dump(pod_data, f)
    print(f"✓ POD data saved to: {pod_path}")
    
    # Level 3: Sensors (DMD residuals sampled at random locations)
    clim_data_sensors, sensor_mask, sensor_idxs, n_sensors = lf_prep.prepare_sensors(
        y_true_data=data,
        y_dmd_pred=y_dmd_pred,
        dmd_years=dmd_years,
        n_sensors=128,
        seed=0,
        sensor_noise_std=0.02  # Add 2% noise to simulate sensor uncertainty
    )
    
    # Split sensor data
    x3_train, x3_val, x3_test = lf_prep.split_sensor_data(
        clim_data_sensors, n_year_train, n_year_val, n_year_test
    )
    
    # ========================================================================
    # PHASE 4: HIGH-FIDELITY TARGET PREPARATION
    # ========================================================================
    # CRITICAL WORKFLOW CONSISTENCY:
    # - Residual statistics (mean/std) computed ONLY on ice mask pixels
    # - Training loss computed ONLY on ice mask pixels
    # - Conformal calibration computed ONLY on ice mask pixels
    # This ensures all components use the same pixel distribution for coherency
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: HIGH-FIDELITY TARGET PREPARATION")
    print("="*80)
    
    hf_prep = HighFidelityTargetPrep(config, data, y_dmd_pred, dmd_years, 
                                     region_mask, n_year_train, n_year_val, n_year_test)
    
    # Compute residuals (full spatial domain)
    y_dmd_residuals, y_true = hf_prep.compute_residuals()
    
    # Split and normalize (extracts ice mask pixels & computes stats on them only)
    y_train, y_val, y_test, train_mean, train_std = \
        hf_prep.split_and_normalize_residuals(y_dmd_residuals)
    
    # Verify region extraction (data already masked during normalization)
    y_train, y_val, y_test, n_pixel_region = \
        hf_prep.extract_region(y_train, y_val, y_test)
    
    # Store residual scaling parameters for conformal calibration
    # NOTE: These statistics are computed ONLY on ice mask pixels (coherent workflow)
    residual_scaler = {
        'mean': train_mean,
        'std': train_std
    }
    
    # ========================================================================
    # PHASE 5: DATA SCALING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: DATA SCALING")
    print("="*80)
    
    scaler = DataScaler(config)
    x1_train, x1_val, x1_test, x2_train, x2_val, x2_test, x3_train, x3_val, x3_test, scalers = \
        scaler.scale_all_levels(x1_train, x1_val, x1_test,
                               x2_train, x2_val, x2_test,
                               x3_train, x3_val, x3_test)
    
    # ========================================================================
    # PHASE 6: PYTORCH DATASET CREATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 6: PYTORCH DATASET CREATION")
    print("="*80)
    
    # Create x0 (time parameter)
    x0 = np.linspace(0, 1, 365)
    x0_train = np.expand_dims(np.stack([x0] * n_year_train, axis=0), axis=-1)
    x0_val = np.expand_dims(np.stack([x0] * n_year_val, axis=0), axis=-1)
    x0_test = np.expand_dims(np.stack([x0] * n_year_test, axis=0), axis=-1)
    
    # Create sequences
    torch_prep = TorchDatasetPreparation(config)
    sequences = torch_prep.create_sequences(
        x0_train, x1_train, x2_train, x3_train, y_train,
        x0_val, x1_val, x2_val, x3_val, y_val,
        x0_test, x1_test, x2_test, x3_test, y_test
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader, train_dataset = \
        torch_prep.create_dataloaders(sequences)
    
    # ... (dopo aver creato i dataloaders e prima di creare il modello) ...

    # ========================================================================
    # MEMORY CLEANUP (CRITICO PER EVITARE OOM)
    # ========================================================================
    print("\n🧹 FREEING RAM BEFORE MODEL CREATION...")
    try:
        # Delete large variables no longer needed for training
        # NOTE: Keeping x, y coordinates for later visualization
        del y_dmd_pred      # ~8.5 GB
        del y_dmd_residuals # ~8.5 GB
        del data            # ~8.5 GB
        # x, y preserved for save_sample_predictions()
        
        # Delete intermediate copies if they exist
        if 'y_true' in locals(): del y_true
        
        # Force garbage collection
        import gc
        gc.collect()
        print("✓ RAM cleaned. Proceeding to Model Setup.")
    except Exception as e:
        print(f"Warning during cleanup: {e}")

    # ========================================================================
    
    # ========================================================================
    # PHASE 7: MODEL SETUP
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 7: MODEL SETUP")
    print("="*80)
    
    # Get actual dimensions from scaled data (after POD and scaling)
    levels_dim = {
        "level_1": x1_train.shape[-1] if x1_train.ndim > 1 else 1,  # POD modes for thickness
        "level_2": x2_train.shape[-1] if x2_train.ndim > 1 else 1,  # POD modes for SST
        "level_3": x3_train.shape[-1] if x3_train.ndim > 1 else 1   # Number of sensors
    }
    
    print(f"Level dimensions after POD and scaling:")
    print(f"  Level 1 (Thickness POD): {levels_dim['level_1']} modes")
    print(f"  Level 2 (SST POD):       {levels_dim['level_2']} modes")
    print(f"  Level 3 (Sensors):       {levels_dim['level_3']} sensors")
    
    model_setup = ModelSetup(config, levels_dim, n_pixel_region, train_dataset)
    model = model_setup.create_model()
    loss_fn = model_setup.create_loss_function()
    optimizer, lr_scheduler = model_setup.create_optimizer_scheduler(model)
    
    # ========================================================================
    # PHASE 8: MODEL TRAINING
    # ========================================================================
    if not args.skip_training:
        print("\n" + "="*80)
        print("PHASE 8: MODEL TRAINING")
        print("="*80)
        
        trainer = ModelTrainer(config, model, loss_fn, optimizer, lr_scheduler,
                              train_loader, val_loader)
        trainer.train()
    else:
        print("\n" + "="*80)
        print("PHASE 8: LOADING EXISTING MODEL")
        print("="*80)
        
        trainer = ModelTrainer(config, model, loss_fn, optimizer, lr_scheduler,
                              train_loader, val_loader)
        trainer.load_checkpoint()
    
    # ========================================================================
    # PHASE 9: CONFORMAL CALIBRATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 9: CONFORMAL CALIBRATION")
    print("="*80)
    
    # Create a smaller batch dataloader for calibration to reduce GPU memory pressure
    from torch.utils.data import DataLoader
    calibration_batch_size = max(1, val_loader.batch_size // 4)  # Use 1/4 of training batch size
    cal_loader = DataLoader(
        val_loader.dataset,
        batch_size=calibration_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print(f"Using calibration batch size: {calibration_batch_size} (reduced from {val_loader.batch_size} for memory safety)")
    print("Note: Batch size does NOT affect calibration quality - results are identical, just slower with smaller batches")
    
    # =========================================================================
    # LOAD DMD FORECASTS (REQUIRED FOR VALIDATION DATA PREPARATION)
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING DMD FORECASTS FOR CALIBRATION")
    print("="*70)
    
    force_dmd_path = "/scratch_global/u10715220/checkpoints/dmd_forecasts_rank5_bootstrap100_years2-34.pkl"
    
    # Safety Check: Only load if variable is not already defined
    if 'y_dmd_pred' not in locals():
        print(f"🔧 Loading DMD forecasts from: {force_dmd_path}")

        if not os.path.exists(force_dmd_path):
            print(f"❌ ERROR: File not found: {force_dmd_path}")
            sys.exit(1)

        with open(force_dmd_path, 'rb') as f:
            dmd_content = dill.load(f)

        # Extract Data (Handling Dictionary vs Array)
        if isinstance(dmd_content, dict) and 'y_pred_mean' in dmd_content:
            y_dmd_pred = dmd_content['y_pred_mean']
            print("   ✅ Extracted 'y_pred_mean' from dictionary.")
        else:
            y_dmd_pred = dmd_content
            print("   ✅ Extracted data directly.")
            
        print(f"   Shape of loaded forecasts: {y_dmd_pred.shape}")
        
        # CRITICAL: Clip DMD predictions to [0,1] to ensure physical constraints
        y_dmd_pred = np.clip(y_dmd_pred, 0, 1)
        print(f"   ✅ DMD predictions clipped to [0,1] range")
        
        # CRITICAL: Crop DMD predictions to match the training data dimensions
        if bbox is not None:
            rmin, rmax = bbox['rmin'], bbox['rmax']
            cmin, cmax = bbox['cmin'], bbox['cmax']
            print(f"   🔧 Cropping DMD from {y_dmd_pred.shape[2:]} to match training data bbox: [{rmin}:{rmax+1}, {cmin}:{cmax+1}]")
            y_dmd_pred = y_dmd_pred[:, :, rmin:rmax+1, cmin:cmax+1]
            print(f"   ✅ DMD cropped to: {y_dmd_pred.shape}")
        else:
            print(f"   ⚠️  Warning: No bbox available, DMD predictions not cropped")
    else:
        print("✓ DMD forecasts already loaded")
    
    # ========================================================================
    # PREPARE DMD AND TRUE SIC FOR VALIDATION (NEEDED FOR CALIBRATION)
    # ========================================================================
    print("\n" + "="*70)
    print("PREPARING VALIDATION DATA FOR CALIBRATION")
    print("="*70)
    
    # Extract DMD validation years
    dmd_val_years = y_dmd_pred[n_year_train:n_year_train + n_year_val]
    dmd_val_continuous = dmd_val_years.reshape(-1, ny, nx)
    val_baseline_active = dmd_val_continuous[:, region_mask]
    val_baseline_active_tensor = torch.tensor(val_baseline_active, dtype=torch.float32)
    
    print(f"✓ DMD validation baseline: {val_baseline_active_tensor.shape}")
    print(f"   Min: {val_baseline_active_tensor.min():.4f}, Max: {val_baseline_active_tensor.max():.4f}")
    
    # Load original true SIC data for validation
    print("\n" + "="*70)
    print("LOADING ORIGINAL TRUE SIC DATA FOR VALIDATION")
    print("="*70)
    
    data_file = config.data_path / "Antarctic_years_1989_2024i.pkl"
    print(f"Loading from: {data_file}")
    
    with open(data_file, 'rb') as f:
        mask_land_val, mask_ice_val, data_val_load, _, _, x_val_load, y_val_load = dill.load(f)
    
    print(f"✓ Loaded {len(data_val_load)} years of data")
    
    # Apply same preprocessing
    thin = config.parameters.get('data', {}).get('thin', 1)
    data_thinned_val, _, _, _, _, _, _ = thin_data(
        thin, data_val_load, None, None, x_val_load, y_val_load, mask_ice_val, mask_land_val
    )
    data_no_leap_val = del_leap(data_thinned_val)
    
    print(f"✓ Applied thinning (factor={thin}) and removed leap days")
    
    # Apply bbox cropping if exists
    if bbox is not None:
        rmin, rmax = bbox['rmin'], bbox['rmax']
        cmin, cmax = bbox['cmin'], bbox['cmax']
        data_no_leap_val = [year_data[:, rmin:rmax+1, cmin:cmax+1] for year_data in data_no_leap_val]
        print(f"✓ Cropped to bbox: {data_no_leap_val[0].shape}")
    
    # Select same years as training (1993-2023, indices [4:-1])
    data_selected_val = data_no_leap_val[4:-1]
    print(f"✓ Selected years 1993-2023: {len(data_selected_val)} years")
    
    # Extract validation years
    y_true_val_years = data_selected_val[n_year_train:n_year_train + n_year_val]
    y_true_val_stacked = np.stack(y_true_val_years, axis=0)  # (n_val_years, 365, ny, nx)
    y_true_val_continuous = y_true_val_stacked.reshape(-1, ny, nx)
    y_true_val_active = y_true_val_continuous[:, region_mask]
    y_true_val_tensor = torch.tensor(y_true_val_active, dtype=torch.float32)
    
    # Clean up validation data loading variables
    del data_val_load, data_thinned_val, data_no_leap_val, data_selected_val
    del y_true_val_years, y_true_val_stacked, y_true_val_continuous, y_true_val_active
    del mask_land_val, mask_ice_val, x_val_load, y_val_load
    gc.collect()
    
    print(f"✓ True SIC validation: {y_true_val_tensor.shape}")
    print(f"   Min: {y_true_val_tensor.min():.4f}, Max: {y_true_val_tensor.max():.4f}")
    
    # Enable spatio-temporal calibration (pixelwise + seasonal)
    calibrator = ConformalCalibration(config, model, pixelwise=True, temporal=True, n_seasons=4)
    q_scores = calibrator.calibrate_conditional(cal_loader, residual_scaler, val_baseline_active_tensor, y_true_val_tensor)
    
    # ========================================================================
    # PHASE 10: TEST EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 10: TEST EVALUATION")
    print("="*80)
    
    evaluator = TestEvaluator(config, model, q_scores, calibrator.scenarios, residual_scaler)
    
    # =========================================================================
    # PREPARE DMD TEST DATA
    # =========================================================================
    # dmd_test_years = y_dmd_pred[n_year_train+n_year_val:]
    # Prepare DMD baseline for test set
    dmd_test_years = y_dmd_pred[n_year_train+n_year_val:]
    
    # Extract spatial dimensions from DMD predictions (should match cropped data after bbox applied)
    ny_dmd, nx_dmd = dmd_test_years.shape[2], dmd_test_years.shape[3]
    print(f"DMD spatial dimensions after cropping: {ny_dmd} x {nx_dmd}, Training data dimensions: {ny} x {nx}")
    
    # Reshape using DMD dimensions
    dmd_test_continuous = dmd_test_years.reshape(-1, ny_dmd, nx_dmd)
    
    # Verify dimensions match after cropping
    if (ny_dmd, nx_dmd) != (ny, nx):
        print(f"❌ ERROR: Dimension mismatch persists after cropping!")
        print(f"   DMD shape: {(ny_dmd, nx_dmd)}, Training data shape: {(ny, nx)}")
        print(f"   region_mask shape: {region_mask.shape}")
        raise ValueError(f"DMD dimensions {(ny_dmd, nx_dmd)} don't match training data {(ny, nx)} even after cropping")
    
    print(f"✓ Dimensions match: {(ny, nx)}")
    
    # Use the same region_mask since dimensions now match
    region_mask_dmd = region_mask
    
    # Verify pixel counts match
    n_pixels_model = region_mask.sum()  # Model was trained on this
    n_pixels_dmd = region_mask_dmd.sum()  # DMD uses this (should be same)
    print(f"Model trained on {n_pixels_model} pixels, DMD baseline has {n_pixels_dmd} pixels in mask")
    
    if n_pixels_model != n_pixels_dmd:
        print(f"❌ ERROR: Pixel count mismatch! Model outputs {n_pixels_model} pixels but DMD provides {n_pixels_dmd}")
        raise ValueError("Cannot proceed: model output size doesn't match DMD baseline size")
    
    test_baseline_active = dmd_test_continuous[:, region_mask_dmd]
    test_baseline_active_tensor = torch.tensor(test_baseline_active, dtype=torch.float32)
    
    # Flatten DMD baseline for full grid evaluation using correct dimensions
    dmd_test_flat = dmd_test_continuous.reshape(-1, ny_dmd * nx_dmd)
    
    # =========================================================================
    # LOAD ORIGINAL TRUE SIC DATA FOR GROUND TRUTH (BEFORE EVALUATION!)
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING ORIGINAL TRUE SIC DATA FOR GROUND TRUTH")
    print("="*70)
    
    # Reload the original data file (same as initial load)
    data_file = config.data_path / "Antarctic_years_1989_2024i.pkl"
    print(f"Loading from: {data_file}")
    
    with open(data_file, 'rb') as f:
        mask_land_reload, mask_ice_reload, data_reload, _, _, x_reload, y_reload = dill.load(f)
    
    print(f"✓ Loaded {len(data_reload)} years of data")
    
    # Apply same preprocessing as initial load
    thin = config.parameters.get('data', {}).get('thin', 1)
    data_thinned, _, _, _, _, _, _ = thin_data(
        thin, data_reload, None, None, x_reload, y_reload, mask_ice_reload, mask_land_reload
    )
    data_no_leap = del_leap(data_thinned)
    
    print(f"✓ Applied thinning (factor={thin}) and removed leap days")
    
    # Apply bbox cropping if exists (same as initial data)
    if bbox is not None:
        rmin, rmax = bbox['rmin'], bbox['rmax']
        cmin, cmax = bbox['cmin'], bbox['cmax']
        data_no_leap = [year_data[:, rmin:rmax+1, cmin:cmax+1] for year_data in data_no_leap]
        print(f"✓ Cropped to bbox: {data_no_leap[0].shape}")
    
    # Select same years as training (years 1993-2023, indices [4:-1])
    data_selected = data_no_leap[4:-1]
    print(f"✓ Selected years 1993-2023: {len(data_selected)} years")
    
    # Extract test year(s)
    y_true_test_years = data_selected[n_year_train + n_year_val:]
    y_true_test_stacked = np.stack(y_true_test_years, axis=0)  # (n_test_years, 365, ny, nx)
    print(f"✓ Test data shape: {y_true_test_stacked.shape}")
    
    # Reshape to continuous time and extract active region
    y_true_test_continuous = y_true_test_stacked.reshape(-1, ny, nx)
    y_true_test_active = y_true_test_continuous[:, region_mask]
    y_true_test_tensor = torch.tensor(y_true_test_active, dtype=torch.float32)
    
    print(f"✓ Ground truth active region: {y_true_test_tensor.shape}")
    print(f"   Min: {y_true_test_tensor.min():.4f}, Max: {y_true_test_tensor.max():.4f}")
    
    # Clean up (but keep y_true_test_tensor for evaluation!)
    del data_reload, data_thinned, data_no_leap, data_selected
    del y_true_test_years, y_true_test_stacked, y_true_test_continuous, y_true_test_active
    del mask_land_reload, mask_ice_reload, x_reload, y_reload
    gc.collect()
    
    # =========================================================================
    # NOW EVALUATE WITH ACTUAL GROUND TRUTH SIC
    # =========================================================================
    print("\n" + "="*70)
    print("EVALUATING MODEL WITH ORIGINAL GROUND TRUTH SIC")
    print("="*70)
    
    # Evaluate - pass actual ground truth SIC
    df_results = evaluator.evaluate_test_physics(test_loader, dmd_test_flat, region_mask_dmd, y_true_test_tensor)
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(df_results.to_string(index=False))
    
    # ========================================================================
    # Ground truth already loaded above, proceed with other results
    # ========================================================================
    # PHASE 10B: GENERATE PREDICTIONS FOR ALL SCENARIOS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 10B: GENERATING PREDICTIONS FOR ALL SCENARIOS")
    print("="*80)
    
    scenario_predictions = {}
    
    for scenario_name, mask_cfg in evaluator.scenarios.items():
        print(f"\n  Processing scenario: {scenario_name}")
        print(f"    Mask configuration: {mask_cfg}")
        
        # Generate predictions with FORCED mask configuration
        # The mask_cfg is passed to generate_full_sic_predictions which will
        # call force_mask() on each batch before model forward pass
        vec_low, vec_med, vec_high, vec_true = evaluator.generate_full_sic_predictions(
            test_loader, test_baseline_active_tensor, q_scores[scenario_name], 
            y_true_test_tensor, mask_config=mask_cfg
        )
        
        scenario_predictions[scenario_name] = (vec_low, vec_med, vec_high, vec_true)
        
        print(f"  ✓ Completed: {scenario_name}")
    
    # ========================================================================
    # PHASE 10C: COMPUTE CLIMATOLOGY FROM TRAIN + VAL DATA
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 10C: COMPUTING CLIMATOLOGICAL BASELINE")
    print("="*80)
    
    # Reload train + val data for climatology
    print("Loading train and validation data for climatology...")
    data_file = config.data_path / "Antarctic_years_1989_2024i.pkl"
    
    with open(data_file, 'rb') as f:
        mask_land_reload, mask_ice_reload, data_reload, _, _, x_reload, y_reload = dill.load(f)
    
    # Apply same preprocessing
    thin = config.parameters.get('data', {}).get('thin', 1)
    data_thinned, _, _, _, _, _, _ = thin_data(
        thin, data_reload, None, None, x_reload, y_reload, mask_ice_reload, mask_land_reload
    )
    data_no_leap = del_leap(data_thinned)
    
    # Apply bbox cropping
    if bbox is not None:
        rmin, rmax = bbox['rmin'], bbox['rmax']
        cmin, cmax = bbox['cmin'], bbox['cmax']
        data_no_leap = [year_data[:, rmin:rmax+1, cmin:cmax+1] for year_data in data_no_leap]
    
    # Select same years (1993-2023)
    data_selected = data_no_leap[4:-1]
    
    # Extract ONLY train years for climatology (not validation!)
    data_train = data_selected[:n_year_train]
    data_train_stacked = np.stack(data_train, axis=0)  # (n_train, 365, ny, nx)
    
    # Compute day-wise average across training years only
    climatology_avg = data_train_stacked.mean(axis=0)  # (365, ny, nx)
    
    print(f"✓ Climatology computed from {n_year_train} training years only")
    print(f"  Shape: {climatology_avg.shape}")
    print(f"  Min: {climatology_avg.min():.4f}, Max: {climatology_avg.max():.4f}")
    
    # Create test climatology baseline (aligned with test data)
    climatology_test = np.tile(climatology_avg, (n_year_test, 1, 1))  # (n_test_years, 365, ny, nx)
    climatology_test_continuous = climatology_test.reshape(-1, ny, nx)  # (n_test_years*365, ny, nx)
    climatology_test_active = climatology_test_continuous[:, region_mask]  # (n_timesteps, n_pixels)
    climatology_test_tensor = torch.tensor(climatology_test_active, dtype=torch.float32)
    
    print(f"✓ Test climatology baseline created: {climatology_test_tensor.shape}")
    
    # Clean up
    del data_reload, data_thinned, data_no_leap, data_selected
    del data_train, data_train_stacked
    del climatology_test, climatology_test_continuous, climatology_test_active
    del mask_land_reload, mask_ice_reload, x_reload, y_reload
    gc.collect()
    
    # ========================================================================
    # PHASE 11: SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 11: SAVING RESULTS")
    print("="*80)
    
    saver = ResultsSaver(config)
    
    # 1. Training curves
    saver.save_training_curves(trainer)
    
    # 2. Test results table
    saver.save_test_results(df_results)
    
    # 3. Original sample prediction - now with multiple years and seasons
    vec_low_3L, vec_med_3L, vec_high_3L, vec_true_3L = scenario_predictions['3L_all']
    
    # Plot multiple test years with different seasons:
    # - If n_year_test=1: plot all 4 seasons for that year
    # - If n_year_test=4: plot one season per year (cycling through)
    year_indices_to_plot = list(range(min(4, n_year_test)))  # Up to 4 years
    saver.save_sample_predictions(
        vec_low_3L, vec_med_3L, vec_high_3L, vec_true_3L,
        y_dmd_pred, region_mask_dmd, mask_land, x, y,
        year_indices=year_indices_to_plot,
        day_indices=[0, 92, 182, 274]  # Summer, Autumn, Winter, Spring  
    )
    
    # 4. NEW: Day 180 predictions for all scenarios and test years
    saver.save_scenario_day_predictions(scenario_predictions, y_dmd_pred, region_mask_dmd, 
                                        mask_land, x, y, n_year_train, n_year_val, 
                                        n_year_test, target_day=180)
    
    # 5. NEW: Seasonal analysis (spatial maps, time series, box plots)
    saver.save_seasonal_analysis(scenario_predictions, y_dmd_pred, climatology_avg,
                                 region_mask_dmd, mask_land, x, y, 
                                 n_year_train, n_year_val, n_year_test)
    
    # 6. NEW: Baseline comparisons (DMD and climatology)
    # Use the SAME baselines as Phase 10 for consistency
    print("\n🔍 Preparing baselines for comparison...")
    print(f"  test_baseline_active_tensor: {test_baseline_active_tensor.shape}")
    print(f"  climatology_test_tensor: {climatology_test_tensor.shape}")
    saver.save_baseline_comparisons(scenario_predictions, test_baseline_active_tensor, climatology_test_tensor)
    
    # 7. Save all numerical data for ALL scenarios (not just 3L_all)
    saver.save_all_data(q_scores, scenario_predictions)
    
    # 8. NEW: IceNet-style metrics for comparison with published benchmarks
    print("\n🎯 Computing IceNet-style metrics for publication comparison...")
    icenet_metrics, df_icenet = saver.save_icenet_metrics(
        scenario_predictions, region_mask_dmd, mask_land, x, y, grid_cell_area_km2=625.0
    )
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "="*80)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"Results saved to: {saver.results_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    if trainer.wandb_run:
        trainer.wandb_run.finish()


if __name__ == "__main__":
    main()
