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
    - Coefficients scaled by singular values for energy weighting
    - POD coefficients NORMALIZED after decomposition for balanced learning
    - region_mask (ALL dynamic ice pixels, probability > 0) used ONLY for target variable (y) loss computation
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
        
        thin = self.config.parameters.get('data', {}).get('thin', 1)  # Default to 1 to keep full 432x432 grid
        
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
        
        return data, data_mean_month, data_mean_week, x, y, mask_ice, mask_land, ny, nx, thickness_data, sst_data
        
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
        DESIRED_TRAIN = 21
        DESIRED_VAL = 5
        DESIRED_TEST = 5
        
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
    
    def __init__(self, config):
        self.config = config
        self.dmd_file = config.checkpoint_dir / "dmd_fits_all_years" / "dmd_forecasts_rank5_bootstrap50_years2-34.pkl"
        
    def load_dmd_predictions(self):
        """Load pre-computed DMD predictions"""
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
                            n_POD=64):
        """Apply POD (Proper Orthogonal Decomposition) to reduce x1 and x2 dimensions
        
        POD is computed on the FULL spatial grid (ny*nx = 432*432) for maximum
        information retention, then reduced to n_POD modes for computational efficiency.
        
        Args:
            x1_train, x1_val, x1_test: Ice thickness data (time_steps, ny*nx)
            x2_train, x2_val, x2_test: SST data (time_steps, ny*nx)
            n_POD: Number of POD modes to retain (default: 64)
            
        Returns:
            Reduced data for all splits, POD basis and singular values
            Data is scaled by singular values for energy weighting
        """
        print("\nAPPLYING POD DECOMPOSITION")
        print("="*70)
        
        # Get dimensions
        N_dof_x1 = x1_train.shape[1]  # Number of spatial pixels for thickness
        N_dof_x2 = x2_train.shape[1]  # Number of spatial pixels for SST
        n_years_train = x1_train.shape[0]
        n_years_val = x1_val.shape[0]
        n_years_test = x1_test.shape[0]
        
        print(f"\nOriginal dimensions:")
        print(f"  Ice Thickness: {N_dof_x1} pixels")
        print(f"  SST: {N_dof_x2} pixels")
        print(f"  POD modes: {n_POD}")
        
        # ====================================================================
        # POD for Level 1 (Ice Thickness)
        # ====================================================================
        print("\n" + "-"*70)
        print("LEVEL 1: ICE THICKNESS POD")
        print("-"*70)
        
        # Reshape for POD: transpose to (n_pixels, n_timesteps)
        x1_train_pod = x1_train.T  # (N_dof_x1, n_timesteps)
        x1_val_pod = x1_val.T
        x1_test_pod = x1_test.T
        
        print(f"POD input shapes: train{x1_train_pod.shape}, val{x1_val_pod.shape}, test{x1_test_pod.shape}")
        
        # Compute POD on training data ONLY (no data leakage)
        print("Computing randomized SVD on training data...")
        U1, S1 = compute_randomized_SVD(x1_train_pod, n_POD, N_dof_x1, 1)
        
        # Explained variance
        energy_captured = np.cumsum(S1) / np.sum(S1)
        print(f"✓ SVD complete:")
        print(f"  Energy captured by {n_POD} modes: {energy_captured[n_POD-1]*100:.2f}%")
        print(f"  Singular values range: [{S1[0]:.2e}, {S1[-1]:.2e}]")
        
        # Project ALL data onto POD basis (fitted on training data only)
        print("Projecting data onto POD basis...")
        x1_train_reduced = np.dot(x1_train_pod.T, U1)  # (n_timesteps, n_POD)
        x1_val_reduced = np.dot(x1_val_pod.T, U1)
        x1_test_reduced = np.dot(x1_test_pod.T, U1)
        
        # Normalize by dividing by singular values (prevents mode dominance)
        print("Normalizing POD coefficients by singular values...")
        x1_train_reduced = x1_train_reduced / S1[:n_POD]
        x1_val_reduced = x1_val_reduced / S1[:n_POD]
        x1_test_reduced = x1_test_reduced / S1[:n_POD]
        
        print(f"✓ Reduced shapes: train{x1_train_reduced.shape}, val{x1_val_reduced.shape}, test{x1_test_reduced.shape}")
        
        # ====================================================================
        # POD for Level 2 (SST)
        # ====================================================================
        print("\n" + "-"*70)
        print("LEVEL 2: SEA SURFACE TEMPERATURE POD")
        print("-"*70)
        
        # Reshape for POD: transpose to (n_pixels, n_timesteps)
        x2_train_pod = x2_train.T  # (N_dof_x2, n_timesteps)
        x2_val_pod = x2_val.T
        x2_test_pod = x2_test.T
        
        print(f"POD input shapes: train{x2_train_pod.shape}, val{x2_val_pod.shape}, test{x2_test_pod.shape}")
        
        # Compute POD on training data ONLY (no data leakage)
        print("Computing randomized SVD on training data...")
        U2, S2 = compute_randomized_SVD(x2_train_pod, n_POD, N_dof_x2, 1)
        
        # Explained variance
        energy_captured = np.cumsum(S2) / np.sum(S2)
        print(f"✓ SVD complete:")
        print(f"  Energy captured by {n_POD} modes: {energy_captured[n_POD-1]*100:.2f}%")
        print(f"  Singular values range: [{S2[0]:.2e}, {S2[-1]:.2e}]")
        
        # Project ALL data onto POD basis (fitted on training data only)
        print("Projecting data onto POD basis...")
        x2_train_reduced = np.dot(x2_train_pod.T, U2)  # (n_timesteps, n_POD)
        x2_val_reduced = np.dot(x2_val_pod.T, U2)
        x2_test_reduced = np.dot(x2_test_pod.T, U2)
        
        # Normalize by dividing by singular values (prevents mode dominance)
        print("Normalizing POD coefficients by singular values...")
        x2_train_reduced = x2_train_reduced / S2[:n_POD]
        x2_val_reduced = x2_val_reduced / S2[:n_POD]
        x2_test_reduced = x2_test_reduced / S2[:n_POD]
        
        
        print(f"✓ Reduced shapes: train{x2_train_reduced.shape}, val{x2_val_reduced.shape}, test{x2_test_reduced.shape}")
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "="*70)
        print("POD REDUCTION SUMMARY")
        print("="*70)
        print(f"Dimension reduction:")
        print(f"  Level 1 (Thickness): {N_dof_x1} → {n_POD} ({n_POD/N_dof_x1*100:.1f}% of original)")
        print(f"  Level 2 (SST):       {N_dof_x2} → {n_POD} ({n_POD/N_dof_x2*100:.1f}% of original)")
        print(f"  Speedup factor:      ~{N_dof_x1/n_POD:.1f}x")
        print("="*70)
        
        # Store POD metadata
        pod_data = {
            'n_POD': n_POD,
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
        
    def prepare_sensors(self, n_sensors=128, ice_prob_threshold=0.5, min_ice_conc=0.1, seed=0, add_noise=True, noise_std=0.05):
        """Create probabilistic sensor placement with 50% ice probability threshold
        
        Uses a stricter threshold (50%) than the region mask (>0%) to ensure sensors
        are placed where ice is consistently present, avoiding mostly-zero timeseries.
        
        Args:
            add_noise: If True, adds Gaussian noise to simulate realistic measurement errors
            noise_std: Standard deviation of noise relative to signal (default: 0.05 = 5% noise)
        """
        print(f"\nPREPARING LEVEL 3: SENSORS (n={n_sensors}, threshold={ice_prob_threshold})")
        print("-"*70)
        if add_noise:
            print(f"  ⚠️  NOISE INJECTION ENABLED: {noise_std*100:.1f}% Gaussian noise (prevents overfitting on perfect observations)")
        
        # Calculate ice probability (same calculation as region mask, different threshold)
        ice_presence = (self.data_tot > min_ice_conc).astype(float)
        ice_probability = ice_presence.mean(axis=(0, 1))
        
        # Create probabilistic mask with STRICTER threshold (50% vs 5% for region)
        mask_ice_probable = (ice_probability >= ice_prob_threshold) & self.mask_ice
        mask_ice_probable_idxs = np.argwhere(mask_ice_probable)
        
        print(f"  Sensor mask threshold: {ice_prob_threshold} (stricter than region mask >0 to ensure consistent ice presence)")
        print(f"  Available sensor locations: {len(mask_ice_probable_idxs)}")
        
        # Random sensor placement
        np.random.seed(seed)
        if len(mask_ice_probable_idxs) >= n_sensors:
            sensor_idxs = mask_ice_probable_idxs[
                np.random.choice(mask_ice_probable_idxs.shape[0], n_sensors, replace=False)
            ]
        else:
            print(f"  ⚠️  Only {len(mask_ice_probable_idxs)} valid locations, using all")
            sensor_idxs = mask_ice_probable_idxs
            n_sensors = len(sensor_idxs)
            
        # Create sensor mask
        sensor_mask = np.zeros_like(self.mask_ice, dtype=bool)
        sensor_mask[tuple(sensor_idxs.T)] = True
        
        clim_data_sensors = self.data_tot[:, :, sensor_mask]
        
        # Add realistic measurement noise to prevent overfitting on perfect observations
        if add_noise:
            np.random.seed(seed)
            # Clip to [0, 1] to maintain physical constraints
            signal_range = clim_data_sensors.max() - clim_data_sensors.min()
            noise = np.random.normal(0, noise_std * signal_range, clim_data_sensors.shape)
            clim_data_sensors_noisy = np.clip(clim_data_sensors + noise, 0, 1)
            
            # Statistics
            snr = np.std(clim_data_sensors) / noise_std / signal_range if signal_range > 0 else np.inf
            print(f"  Noise statistics:")
            print(f"    Noise std: {noise_std*100:.1f}% of signal range")
            print(f"    Signal-to-Noise Ratio (SNR): {snr:.2f}")
            print(f"    Mean absolute noise: {np.abs(noise).mean():.6f}")
            
            clim_data_sensors = clim_data_sensors_noisy
        
        print(f"✓ Sensors placed: {n_sensors}")
        print(f"  Data shape: {clim_data_sensors.shape}")
        
        return clim_data_sensors, sensor_mask, sensor_idxs, n_sensors
        
    def split_sensor_data(self, clim_data_sensors, n_year_train, n_year_val, n_year_test):
        """Split sensor data into train/val/test"""
        print("\nSPLITTING SENSOR DATA")
        print("-"*70)
        
        x3_train = clim_data_sensors[:n_year_train]
        x3_val = clim_data_sensors[n_year_train:n_year_train + n_year_val]
        x3_test = clim_data_sensors[n_year_train + n_year_val:n_year_train + n_year_val + n_year_test]
        
        # Flatten time dimension
        x3_train = x3_train.reshape(-1, x3_train.shape[-1])
        x3_val = x3_val.reshape(-1, x3_val.shape[-1])
        x3_test = x3_test.reshape(-1, x3_test.shape[-1])
        
        print(f"✓ Level 3 (Sensors): Train{x3_train.shape}, Val{x3_val.shape}, Test{x3_test.shape}")
        print(f"  Train range: [{x3_train.min():.4f}, {x3_train.max():.4f}]")
        print(f"  Val range:   [{x3_val.min():.4f}, {x3_val.max():.4f}]")
        
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
        """Split residuals and normalize"""
        print("\nSPLITTING AND NORMALIZING RESIDUALS")
        print("-"*70)
        
        # Split
        residuals_train = y_dmd_residuals[:self.n_year_train]
        residuals_val = y_dmd_residuals[self.n_year_train:self.n_year_train + self.n_year_val]
        residuals_test = y_dmd_residuals[self.n_year_train + self.n_year_val:self.n_year_train + self.n_year_val + self.n_year_test]
        
        print(f"✓ Split residuals:")
        print(f"  Train: {residuals_train.shape}")
        print(f"  Val:   {residuals_val.shape}")
        print(f"  Test:  {residuals_test.shape}")
        
        # Normalize (using train statistics)
        train_mean = residuals_train.mean()
        train_std = residuals_train.std()
        
        y_train = (residuals_train - train_mean) / train_std
        y_val = (residuals_val - train_mean) / train_std
        y_test = (residuals_test - train_mean) / train_std
        
        print(f"\n✓ Normalized residuals:")
        print(f"  Train mean/std: {y_train.mean():.6f} / {y_train.std():.6f}")
        print(f"  Val mean/std:   {y_val.mean():.6f} / {y_val.std():.6f}")
        print(f"  Test mean/std:  {y_test.mean():.6f} / {y_test.std():.6f}")
        
        return y_train, y_val, y_test, train_mean, train_std
        
    def extract_region(self, y_train, y_val, y_test):
        """Extract only the pixels in the target region"""
        print("\nEXTRACTING TARGET REGION")
        print("-"*70)
        
        # Define region mask (circular or custom)
        center_x, center_y = 0.01, 0.025
        radius1 = 0.2
        radius2 = 0.3
        
        # This uses the ice probability mask (simplified version)
        # In practice, use the same region_mask from low-fidelity prep
        
        y_train_region = y_train[:, :, self.region_mask]
        y_val_region = y_val[:, :, self.region_mask]
        y_test_region = y_test[:, :, self.region_mask]
        
        n_pixel_region = y_train_region.shape[2]
        
        print(f"✓ Extracted region:")
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
        
        Note: x1 and x2 are POD coefficients already normalized after decomposition,
        so they do NOT need additional robust scaling. Only x3 (sensors) is scaled.
        """
        print("\nSCALING FIDELITY LEVELS")
        print("-"*70)
        
        print("Level 1 (Ice Thickness POD): SKIPPED (already normalized after decomposition)")
        med1, iqr1 = None, None
        
        print("Level 2 (SST POD): SKIPPED (already normalized after decomposition)")
        med2, iqr2 = None, None
        
        print("Level 3 (Sensors): Applying robust scaling...")
        x3_train, x3_val, x3_test, med3, iqr3 = self.robust_scale_fit_transform(
            x3_train, x3_val, x3_test
        )
        
        # Store scalers (x1 and x2 are normalized after decomposition)
        scalers = {
            'level_1': {'median': med1, 'iqr': iqr1, 'method': 'POD_normalized'},
            'level_2': {'median': med2, 'iqr': iqr2, 'method': 'POD_normalized'},
            'level_3': {'median': med3, 'iqr': iqr3, 'method': 'robust_scaler'}
        }
        
        # Save scalers
        scaler_path = self.config.scaler_dir / 'scalers_multifidelity.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
            
        print(f"\n✓ Scaling complete. Scalers saved to: {scaler_path}")
        
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
        train_dataset = MultiFidelityDataset(
            train_features, y_train, device='cpu',
            sequential=self.config.parameters["data"]["sequential_mask"]
        )
        
        val_dataset = MultiFidelityDataset(
            val_features, y_val, device='cpu',
            sequential=False
        )
        
        test_dataset = MultiFidelityDataset(
            test_features, y_test, device='cpu',
            sequential=True
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
        print(f"✓ Model created")
        print(f"  Parameters: {n_params:,}")
        print(f"  Output dimension: {output_dim}")
        print(f"  Quantiles: {self.quantiles}")
        
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
                    err = targets - preds_r[:, :, i, :]
                    loss += torch.mean(torch.max((tau - 1) * err, tau * err))
                    
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
        patience = 20
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
                        
                # Logging
                key_patterns = ["3L_all", "1L_1", "1L_3"]
                fid_msg = " | ".join([
                    f"{p}: T{tr_fid.get(p, 0):.4f}/V{val_fid.get(p, 0):.4f}"
                    for p in key_patterns
                    if tr_fid.get(p) is not None and val_fid.get(p) is not None
                ])
                print(f"Epoch {epoch+1}/{epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f} | {fid_msg}")
                
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
                
                # Save latest checkpoint
                torch.save(self.model.state_dict(), self.latest_path)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    
                    if self.config.parameters["model"]["save_model"]:
                        torch.save(best_model_state, self.checkpoint_path)
                        print(f"  ✓ New best model saved (Val Loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n🛑 Early stopping at epoch {epoch+1}")
                        break
                        
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            
        finally:
            # Restore best weights
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                print("✓ Best model weights restored")
                
            self.save_history()
            print(f"✓ Training state saved to: {self.history_path}")
            
        self.model.eval()
        print(f"\n✅ Training complete. Best Val Loss: {self.best_val_loss:.4f}")


# ============================================================================
# PART 5 COMPLETE
# ============================================================================


class ConformalCalibration:
    """Conformal Quantile Regression calibration"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.target_coverage = 0.90
        self.alpha = 1.0 - self.target_coverage
        
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
    def force_mask(batch, mask_list, device):
        """Force a specific mask configuration on a batch"""
        B = batch['level_0'].shape[0]
        mask_tensor = torch.tensor(mask_list, dtype=torch.bool, device=device)
        batch['mask'] = mask_tensor.unsqueeze(0).expand(B, -1)
        return batch
        
    def calibrate_conditional(self, val_loader):
        """Calibrate Q-scores for each scenario"""
        print("\n" + "="*70)
        print("CONFORMAL CALIBRATION")
        print("="*70)
        print(f"Target Coverage: {self.target_coverage:.0%}")
        
        self.model.eval()
        q_dict = {}
        
        for name, mask_cfg in self.scenarios.items():
            all_scores = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Calibrating {name}"):
                    # Prepare inputs
                    batch_input = {k: v.to(self.config.device) for k, v in batch.items() if k != 'target'}
                    target = batch['target'].to(self.config.device)
                    
                    # Force scenario mask
                    self.force_mask(batch_input, mask_cfg, self.config.device)
                    
                    # Forward pass
                    out = self.model(batch_input)
                    
                    # Reshape to (B, Seq, 3, Pixels)
                    B, S, _ = out.shape
                    n_pixels = out.shape[-1] // 3
                    preds = out.view(B, S, 3, n_pixels)
                    
                    # Compute conformity scores: E = max(low - y, y - high)
                    y_flat = target.flatten()
                    low_flat = preds[:, :, 0, :].flatten()
                    high_flat = preds[:, :, 2, :].flatten()
                    
                    scores = torch.max(low_flat - y_flat, y_flat - high_flat)
                    all_scores.append(scores.cpu().numpy())
                    
            # Compute Q-hat (quantile)
            all_scores = np.concatenate(all_scores)
            q_hat = np.quantile(all_scores, 1 - self.alpha, method='higher')
            q_dict[name] = q_hat
            
            print(f"  {name:12s} → Q-Score: {q_hat:.5f}")
            
        print("✓ Calibration complete")
        return q_dict


class TestEvaluator:
    """Evaluate model on test set with physical metrics"""
    
    def __init__(self, config, model, q_scores, scenarios):
        self.config = config
        self.model = model
        self.q_scores = q_scores
        self.scenarios = scenarios
        
    @staticmethod
    def force_mask(batch, mask_list, device):
        """Force a specific mask configuration"""
        B = batch['level_0'].shape[0]
        mask_tensor = torch.tensor(mask_list, dtype=torch.bool, device=device)
        batch['mask'] = mask_tensor.unsqueeze(0).expand(B, -1)
        return batch
        
    def evaluate_test_physics(self, test_loader, dmd_baseline, region_mask):
        """
        Evaluate on test set comparing Hybrid (DMD + Model) vs DMD baseline
        
        Args:
            test_loader: Test dataloader
            dmd_baseline: (TotalTimeTest, nx*ny) - DMD predictions on full grid
            region_mask: (ny, nx) - Boolean mask for active region
        """
        print("\n" + "="*70)
        print("TEST EVALUATION - PHYSICAL METRICS")
        print("="*70)
        
        self.model.eval()
        results = []
        mask_flat = region_mask.reshape(-1)
        
        for name, mask_cfg in self.scenarios.items():
            q = self.q_scores[name]
            
            total_mae_model = 0
            total_mae_dmd = 0
            covered_count = 0
            total_count = 0
            total_width = 0
            
            current_time_idx = 0
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Evaluating {name}"):
                    # Predict residuals
                    batch_input = {k: v.to(self.config.device) for k, v in batch.items() if k != 'target'}
                    self.force_mask(batch_input, mask_cfg, self.config.device)
                    out = self.model(batch_input)
                    
                    B, S, _ = out.shape
                    n_pixels = out.shape[-1] // 3
                    
                    # Reshape and calibrate
                    out = out.view(B, S, 3, n_pixels).cpu()
                    res_low = out[:, :, 0, :] - q
                    res_med = out[:, :, 1, :]
                    res_high = out[:, :, 2, :] + q
                    
                    # True residuals
                    res_true = batch['target'].view(B, S, n_pixels).cpu()
                    
                    # Get DMD baseline for this time chunk
                    end_idx = current_time_idx + S
                    base_chunk = dmd_baseline[current_time_idx:end_idx, mask_flat]
                    base_tensor = torch.tensor(base_chunk, dtype=torch.float32).unsqueeze(0)
                    
                    # Reconstruct SIC (clamp to [0, 1])
                    sic_pred = torch.clamp(base_tensor + res_med, 0, 1)
                    sic_low = torch.clamp(base_tensor + res_low, 0, 1)
                    sic_high = torch.clamp(base_tensor + res_high, 0, 1)
                    sic_true = torch.clamp(base_tensor + res_true, 0, 1)
                    sic_dmd = torch.clamp(base_tensor, 0, 1)
                    
                    # Compute metrics
                    total_mae_model += torch.abs(sic_true - sic_pred).sum().item()
                    total_mae_dmd += torch.abs(sic_true - sic_dmd).sum().item()
                    
                    # Coverage
                    is_covered = (sic_true >= sic_low) & (sic_true <= sic_high)
                    covered_count += is_covered.sum().item()
                    total_count += is_covered.numel()
                    total_width += (sic_high - sic_low).sum().item()
                    
                    current_time_idx += S
                    
            # Aggregate metrics
            mae_model = total_mae_model / total_count
            mae_dmd = total_mae_dmd / total_count
            improvement = (mae_dmd - mae_model) / (mae_dmd + 1e-6) * 100
            coverage = covered_count / total_count
            width = total_width / total_count
            
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
        
    def generate_full_sic_predictions(self, test_loader, baseline_masked, q_score):
        """
        Generate full SIC predictions for visualization
        
        Args:
            test_loader: Test dataloader
            baseline_masked: (TotalTime, n_pixels_region) - DMD baseline on active region
            q_score: Float - Calibration correction
            
        Returns:
            Tuple of (low, median, high, true) predictions
        """
        print("\nGENERATING FULL SIC PREDICTIONS")
        print("-"*70)
        
        self.model.eval()
        l_low, l_med, l_high, l_true = [], [], [], []
        curr_t = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                # Input
                batch_input = {k: v.to(self.config.device) for k, v in batch.items() if k != 'target'}
                res_true = batch['target'].to(self.config.device)
                
                # Predict
                out = self.model(batch_input)
                B, S, _ = out.shape
                n_pixels = out.shape[-1] // 3
                
                # Reshape and calibrate
                preds = out.view(B, S, 3, n_pixels)
                r_low = preds[:, :, 0, :] - q_score
                r_med = preds[:, :, 1, :]
                r_high = preds[:, :, 2, :] + q_score
                
                # Get baseline
                end_t = curr_t + S
                base_slice = baseline_masked[curr_t:end_t].to(self.config.device)
                base_tensor = base_slice.unsqueeze(0)
                
                # Reconstruct and clamp
                s_med = torch.clamp(base_tensor + r_med, 0, 1)
                s_low = torch.clamp(base_tensor + r_low, 0, 1)
                s_high = torch.clamp(base_tensor + r_high, 0, 1)
                s_true = torch.clamp(base_tensor + res_true, 0, 1)
                
                # Accumulate
                l_med.append(s_med.cpu())
                l_low.append(s_low.cpu())
                l_high.append(s_high.cpu())
                l_true.append(s_true.cpu())
                
                curr_t += S
                
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
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Overall losses
        axes[0, 0].plot([max(eps, float(v)) for v in trainer.train_losses], 
                       label='Train Loss', color='blue')
        axes[0, 0].plot([max(eps, float(v)) for v in trainer.val_losses], 
                       label='Val Loss', color='red')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Overall Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (log scale)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, which='both', alpha=0.3)
        
        # Single level performance (train)
        single_patterns = ['1L_1', '1L_2', '1L_3']
        colors_single = {'1L_1': 'green', '1L_2': 'orange', '1L_3': 'purple'}
        axes[0, 1].set_title('Single Level Loss (Train)')
        for pattern in single_patterns:
            if trainer.train_fidelity_losses[pattern]:
                loss_values = [max(eps, float(v)) for v in trainer.train_fidelity_losses[pattern]]
                axes[0, 1].plot(loss_values, label=f'Train {pattern}', color=colors_single[pattern])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (log scale)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, which='both', alpha=0.3)
        
        # Single level performance (val)
        axes[0, 2].set_title('Single Level Loss (Val)')
        for pattern in single_patterns:
            if trainer.val_fidelity_losses[pattern]:
                loss_values = [max(eps, float(v)) for v in trainer.val_fidelity_losses[pattern]]
                axes[0, 2].plot(loss_values, label=f'Val {pattern}', 
                              color=colors_single[pattern], linestyle='--')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss (log scale)')
        axes[0, 2].set_yscale('log')
        axes[0, 2].legend()
        axes[0, 2].grid(True, which='both', alpha=0.3)
        
        # Two level combinations
        two_patterns = ['2L_1+2', '2L_1+3', '2L_2+3']
        colors_two = {'2L_1+2': 'cyan', '2L_1+3': 'magenta', '2L_2+3': 'yellow'}
        axes[1, 0].set_title('Two Level Combinations Loss')
        for pattern in two_patterns:
            if trainer.train_fidelity_losses[pattern]:
                loss_values = [max(eps, float(v)) for v in trainer.train_fidelity_losses[pattern]]
                axes[1, 0].plot(loss_values, label=f'Train {pattern}', 
                              color=colors_two[pattern])
            if trainer.val_fidelity_losses[pattern]:
                loss_values = [max(eps, float(v)) for v in trainer.val_fidelity_losses[pattern]]
                axes[1, 0].plot(loss_values, label=f'Val {pattern}', 
                              color=colors_two[pattern], linestyle='--')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (log scale)')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, which='both', alpha=0.3)
        
        # Best performers comparison
        axes[1, 1].set_title('Best Performers Comparison')
        if trainer.train_fidelity_losses['3L_all']:
            loss_values = [max(eps, float(v)) for v in trainer.train_fidelity_losses['3L_all']]
            axes[1, 1].plot(loss_values, label='Train 3L_all', color='red', linewidth=2)
        if trainer.val_fidelity_losses['3L_all']:
            loss_values = [max(eps, float(v)) for v in trainer.val_fidelity_losses['3L_all']]
            axes[1, 1].plot(loss_values, label='Val 3L_all', 
                          color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (log scale)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, which='both', alpha=0.3)
        
        # Final comparison bar chart
        all_patterns = ["3L_all", "2L_1+2", "2L_1+3", "2L_2+3", "1L_1", "1L_2", "1L_3"]
        final_train = [max(eps, trainer.train_fidelity_losses[p][-1]) if trainer.train_fidelity_losses[p] else eps for p in all_patterns]
        final_val = [max(eps, trainer.val_fidelity_losses[p][-1]) if trainer.val_fidelity_losses[p] else eps for p in all_patterns]
        
        x_pos = np.arange(len(all_patterns))
        width = 0.35
        axes[1, 2].bar(x_pos - width/2, final_train, width, label='Train Loss', alpha=0.8)
        axes[1, 2].bar(x_pos + width/2, final_val, width, label='Val Loss', alpha=0.8)
        axes[1, 2].set_xlabel('Fidelity Pattern')
        axes[1, 2].set_ylabel('Loss (log scale)')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(all_patterns, rotation=45, ha='right')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(True, which='both', alpha=0.3)
        
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
                                year_idx=0, day_idx=180):
        """Save sample prediction visualizations"""
        print(f"\nSaving sample predictions (Year {year_idx}, Day {day_idx})...")
        
        # Extract single day
        true_seq = vec_true[0, day_idx].detach().cpu().numpy()
        pred_seq = vec_med[0, day_idx].detach().cpu().numpy()
        low_seq = vec_low[0, day_idx].detach().cpu().numpy()
        high_seq = vec_high[0, day_idx].detach().cpu().numpy()
        
        # Reconstruct on full grid
        ny, nx = mask_land.shape
        
        # Initialize with DMD baseline
        dmd_day = y_dmd_pred[year_idx, day_idx]
        pred_full = dmd_day.copy()
        low_full = dmd_day.copy()
        high_full = dmd_day.copy()
        true_full = dmd_day.copy()
        
        # Fill in predicted region
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
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        extent = [x.min(), x.max(), y.min(), y.max()]
        
        # True
        im0 = axes[0, 0].imshow(true_full, origin='lower', cmap='viridis', 
                                extent=extent, vmin=0, vmax=1)
        axes[0, 0].set_title(f'Ground Truth (Day {day_idx})')
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
        save_path = self.results_dir / f"predictions_year{year_idx}_day{day_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Sample predictions saved to: {save_path}")
        
    def save_all_data(self, q_scores, vec_low, vec_med, vec_high, vec_true):
        """Save all numerical data for post-processing (with final [0,1] clipping)"""
        print("\nSaving all numerical data...")
        
        # Ensure all predictions are strictly in [0,1] before saving
        data_dict = {
            'q_scores': q_scores,
            'predictions': {
                'low': torch.clamp(vec_low, 0, 1).numpy(),
                'median': torch.clamp(vec_med, 0, 1).numpy(),
                'high': torch.clamp(vec_high, 0, 1).numpy(),
                'true': torch.clamp(vec_true, 0, 1).numpy()
            }
        }
        
        save_path = self.results_dir / "all_predictions.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
            
        print(f"✓ All numerical data saved to: {save_path}")


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
    parser.add_argument("--experiment_number", type=int, default=4,
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
    data, data_mean_month, data_mean_week, x, y, mask_ice, mask_land, ny, nx, thickness_data, sst_data = \
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
    
    # Set split configuration to match c_DMD.py (years 2-34 → 31 total years)
    n_years_total = 31  # 1991-2021
    
    n_year_train = 21
    n_year_val = 5
    n_year_test = 5
    
    # Verification
    current_total = n_year_train + n_year_val + n_year_test
    if current_total != n_years_total:
        print(f"⚠️  WARNING: Split sum ({current_total}) != Available Years ({n_years_total})")
        
    print(f"\n🔧 Split Configuration:")
    print(f"   Train: {n_year_train} years")
    print(f"   Val:   {n_year_val} years")
    print(f"   Test:  {n_year_test} years")
    
    # =========================================================================
    # LOAD PRE-COMPUTED DMD FORECASTS (ALREADY RECONSTRUCTED)
    # =========================================================================
    print("\n🔍 LOADING PRE-COMPUTED DMD FORECASTS...")
    
    # Path to pre-computed forecasts (already includes full reconstruction from c_DMD.py)
    forecasts_file = Path("/scratch_global/u10715220/checkpoints/dmd_forecasts_rank5_bootstrap100_years2-34.pkl")
    
    # Validate file exists
    if not forecasts_file.exists():
        raise FileNotFoundError(f"❌ CRITICAL: Missing DMD forecasts file: {forecasts_file}")
    
    print(f"✓ Forecasts file found: {forecasts_file}")
    print(f"  Loading pre-reconstructed spatial fields...")
    
    # Load pre-computed forecasts (contains fully reconstructed spatial fields)
    with open(forecasts_file, 'rb') as f:
        forecast_data = dill.load(f)
    
    # Extract the reconstructed predictions (already computed by c_DMD.py)
    y_dmd_pred = forecast_data['y_pred_mean']  # Shape: (n_years, 365, ny, nx) - ALREADY RECONSTRUCTED
    dmd_years = forecast_data['years']
    ny, nx = y_dmd_pred.shape[2], y_dmd_pred.shape[3]
    
    # CRITICAL: Clip DMD predictions to [0,1] to ensure physical constraints
    # Without this, residuals can exceed [-1,1] range
    y_dmd_pred = np.clip(y_dmd_pred, 0, 1)
    print(f"✓ DMD predictions clipped to [0,1] range")
    
    print(f"✓ DMD forecasts loaded successfully (pre-reconstructed by c_DMD.py)")
    print(f"  Shape: {y_dmd_pred.shape}")
    print(f"  Years: {dmd_years[0]} to {dmd_years[-1]} ({len(dmd_years)} years)")
    print(f"  Spatial grid: {ny} × {nx}")
    print(f"  Value range: [{y_dmd_pred.min():.4f}, {y_dmd_pred.max():.4f}] (should be [0,1])")
    print(f"  Memory saved: ~10 GB (no reconstruction needed)")

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
    # IMPORTANT: x1 (thickness) and x2 (SST) use the FULL 432x432 grid for POD decomposition,
    #            NOT the region_mask. This maximizes information retention before dimension reduction.
    # 
    # CLIPPING: All final predictions (baseline + residuals) are clipped to [0,1] to maintain probability constraints
    print("\nDEFINING TARGET REGION (PROBABILISTIC ICE MASK - ALL ICE PIXELS)")
    print("-"*70)
    
    ice_probability_threshold = 0.0  # Include ALL pixels where ice ever appears (probability > 0)
    min_ice_concentration = 0.1  # Minimum concentration to consider as "ice present"
    
    # Calculate probability of ice presence for each pixel across all years and days
    ice_presence = (data_tot > min_ice_concentration).astype(float)  # Shape: (n_years, 365, ny, nx)
    ice_probability = ice_presence.mean(axis=(0, 1))  # Average over years and days
    
    # Create mask for regions with ANY ice probability (includes full dynamic ice extent)
    region_mask = (ice_probability > ice_probability_threshold) & mask_ice
    
    n_pixel_region = region_mask.sum()
    print(f"✓ Region mask defined (threshold: {ice_probability_threshold} - ALL dynamic ice pixels)")
    print(f"  Pixels in region: {n_pixel_region} (full dynamic ice extent)")
    print(f"  Original ice mask had: {mask_ice.sum()} pixels")
    print(f"  Total grid pixels: {ny*nx} pixels")
    print(f"  Ice mask coverage of grid: {mask_ice.sum()/(ny*nx)*100:.1f}%")
    print(f"  Region coverage of ice mask: {n_pixel_region / mask_ice.sum()*100:.1f}%")
    
    # Validation check
    if mask_ice.sum() < (ny*nx) * 0.2:  # Less than 20% of grid
        print(f"  ⚠️  WARNING: Ice mask covers only {mask_ice.sum()/(ny*nx)*100:.1f}% of grid.")
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
    
    # Apply POD reduction to speed up computation (64 modes instead of 432*432 = 186,624 pixels)
    # POD is computed on the FULL grid for maximum information, then reduced to 64 modes
    x1_train, x1_val, x1_test, x2_train, x2_val, x2_test, pod_data = \
        lf_prep.apply_pod_reduction(
            x1_train, x1_val, x1_test,
            x2_train, x2_val, x2_test,
            n_POD=64
        )
    
    # Save POD data for later reconstruction if needed
    pod_path = config.checkpoint_dir / 'pod_data.pkl'
    with open(pod_path, 'wb') as f:
        pickle.dump(pod_data, f)
    print(f"✓ POD data saved to: {pod_path}")
    
    # Level 3: Sensors (with increased noise to prevent overfitting)
    clim_data_sensors, sensor_mask, sensor_idxs, n_sensors = lf_prep.prepare_sensors(
        noise_std=0.15  # Increased from 0.05 to 0.15 (15% noise) to prevent overfitting
    )
    
    # Split sensor data
    x3_train, x3_val, x3_test = lf_prep.split_sensor_data(
        clim_data_sensors, n_year_train, n_year_val, n_year_test
    )
    
    # ========================================================================
    # PHASE 4: HIGH-FIDELITY TARGET PREPARATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: HIGH-FIDELITY TARGET PREPARATION")
    print("="*80)
    
    hf_prep = HighFidelityTargetPrep(config, data, y_dmd_pred, dmd_years, 
                                     region_mask, n_year_train, n_year_val, n_year_test)
    
    # Compute residuals
    y_dmd_residuals, y_true = hf_prep.compute_residuals()
    
    # Split and normalize
    y_train, y_val, y_test, train_mean, train_std = \
        hf_prep.split_and_normalize_residuals(y_dmd_residuals)
    
    # Extract region
    y_train, y_val, y_test, n_pixel_region = \
        hf_prep.extract_region(y_train, y_val, y_test)
    
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
    
    levels_dim = {
        "level_1": x1_train.shape[-1] if x1_train.ndim > 1 else 1,
        "level_2": x2_train.shape[-1] if x2_train.ndim > 1 else 1,
        "level_3": x3_train.shape[-1] if x3_train.ndim > 1 else 1
    }
    
    print(f"Level dimensions: {levels_dim}")
    
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
    
    calibrator = ConformalCalibration(config, model)
    q_scores = calibrator.calibrate_conditional(val_loader)
    
    # ========================================================================
    # PHASE 10: TEST EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 10: TEST EVALUATION")
    print("="*80)
    
    evaluator = TestEvaluator(config, model, q_scores, calibrator.scenarios)
    

    # =========================================================================
    # FIX: FORCE-LOAD DMD FORECASTS TO RESOLVE UnboundLocalError
    # =========================================================================
    # Note: os, dill, sys already imported at module level - no need to re-import

    # 1. Path verified in your previous screenshot
    force_dmd_path = "/scratch_global/u10715220/checkpoints/dmd_forecasts_rank5_bootstrap100_years2-34.pkl"
    
    # 2. Safety Check: Only load if variable is not already defined
    if 'y_dmd_pred' not in locals():
        print(f"🔧 Variable 'y_dmd_pred' unbound. Force-loading from: {force_dmd_path}")

        if not os.path.exists(force_dmd_path):
            print(f"❌ ERROR: File not found: {force_dmd_path}")
            sys.exit(1)

        with open(force_dmd_path, 'rb') as f:
            dmd_content = dill.load(f)

        # 3. Extract Data (Handling Dictionary vs Array)
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

    # =========================================================================
    # ORIGINAL CODE RESUMES HERE (Line 2191)
    # =========================================================================
    # dmd_test_years = y_dmd_pred[n_year_train+n_year_val:]
    # Prepare DMD baseline for test set
    dmd_test_years = y_dmd_pred[n_year_train+n_year_val:]
    dmd_test_continuous = dmd_test_years.reshape(-1, ny, nx)
    test_baseline_active = dmd_test_continuous[:, region_mask]
    test_baseline_active_tensor = torch.tensor(test_baseline_active, dtype=torch.float32)
    
    # Flatten DMD baseline for full grid evaluation
    dmd_test_flat = dmd_test_continuous.reshape(-1, ny*nx)
    
    # Evaluate
    df_results = evaluator.evaluate_test_physics(test_loader, dmd_test_flat, region_mask)
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(df_results.to_string(index=False))
    
    # Generate full predictions
    vec_low, vec_med, vec_high, vec_true = evaluator.generate_full_sic_predictions(
        test_loader, test_baseline_active_tensor, q_scores['3L_all']
    )
    
    # ========================================================================
    # PHASE 11: SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 11: SAVING RESULTS")
    print("="*80)
    
    saver = ResultsSaver(config)
    saver.save_training_curves(trainer)
    saver.save_test_results(df_results)
    saver.save_sample_predictions(vec_low, vec_med, vec_high, vec_true,
                                  y_dmd_pred, region_mask, mask_land, x, y,
                                  year_idx=n_year_train+n_year_val, day_idx=180)
    saver.save_all_data(q_scores, vec_low, vec_med, vec_high, vec_true)
    
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
