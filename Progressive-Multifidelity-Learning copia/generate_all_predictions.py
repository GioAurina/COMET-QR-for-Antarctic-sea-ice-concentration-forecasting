#!/usr/bin/env python3
"""
Generate predictions for all scenarios using a trained model.

This script loads a pre-trained model and generates predictions for all 7 scenarios
without retraining. Reuses the exact configuration from training.

Usage:
    python generate_all_predictions.py --experiment_name ice --experiment_number 21

Requirements:
    - Model checkpoint (.pt file)
    - Configuration file (21.yaml)
    - Data file (Antarctic_years_1989_2024i.pkl)
    - DMD forecasts (dmd_forecasts_rank5_bootstrap100_years2-34.pkl)
"""

import os
import sys
import argparse
from pathlib import Path
import dill
import pickle
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader

# Import from main script - use EXACT same imports and workflow
from mf_res_val_cal_gpu import (
    Config, DataLoader_Ice, DMDBaseline, TorchDatasetPreparation,
    ConformalCalibration, TestEvaluator, 
    ModelSetup, LowFidelityDataPrep, HighFidelityTargetPrep, DataScaler
)
from multifidelity_transformer.utils.data import MultiFidelityDataset


def create_args_namespace(base_path, project_path, experiment_name, experiment_number, device="auto"):
    """Create argument namespace exactly as main script expects"""
    class Args:
        def __init__(self):
            self.base_path = str(base_path)
            self.project_path = str(project_path)
            self.scratch_path = "/scratch_global/u10715220"
            self.experiment_name = experiment_name
            self.experiment_number = experiment_number
            self.device = device
            self.no_wandb = True
            self.skip_training = True
            self.output_dir = None
    
    return Args()


def main():
    print("="*80)
    print("GENERATE ALL SCENARIO PREDICTIONS FROM TRAINED MODEL")
    print("="*80)
    
    # ========================================================================
    # PARSE ARGUMENTS
    # ========================================================================
    parser = argparse.ArgumentParser(description="Generate predictions for all scenarios")
    parser.add_argument("--base_path", type=str,
                       default="/work/u10715220",
                       help="Base path where data and checkpoints are stored")
    parser.add_argument("--project_path", type=str,
                       default=str(Path(".").resolve()),
                       help="Path to project code directory")
    parser.add_argument("--experiment_name", type=str, default="ice",
                       help="Experiment name (folder in experiment_configurations)")
    parser.add_argument("--experiment_number", type=int, default=21,
                       help="Experiment number (yaml filename)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="Device to use for computation")
    
    user_args = parser.parse_args()
    base_path = Path(user_args.base_path).resolve()
    project_path = Path(user_args.project_path).resolve()
    
    print(f"\n📂 Base path (data): {base_path}")
    print(f"📂 Project path (code): {project_path}")
    print(f"📝 Experiment: {user_args.experiment_name}_{user_args.experiment_number}")
    
    # ========================================================================
    # LOAD CONFIGURATION - Exactly as main script does
    # ========================================================================
    print("\n📋 Loading configuration from YAML file...")
    
    # Create args namespace that Config class expects
    args = create_args_namespace(
        base_path,
        project_path,
        user_args.experiment_name, 
        user_args.experiment_number,
        user_args.device
    )
    
    # Initialize Config - this will load the yaml file
    config = Config(args)
    
    # Verify model checkpoint exists
    checkpoint_path = config.checkpoint_dir / f"{config.experiment}.pt"
    if not checkpoint_path.exists():
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        print(f"   Expected at: {config.checkpoint_dir}")
        sys.exit(1)
    
    print(f"✓ Configuration loaded successfully")
    print(f"   Experiment: {config.experiment}")
    print(f"   Device: {config.device}")
    print(f"   Thin parameter: {config.thin}")
    print(f"   Model checkpoint: {checkpoint_path}")
    
    # ========================================================================
    # LOAD AND PREPROCESS DATA - Exactly as main script does
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: DATA LOADING AND PREPROCESSING")
    print("="*80)
    
    data_loader = DataLoader_Ice(config)
    
    # Load raw data
    mask_land_, mask_ice_, data_, data_mean_month_, data_mean_week_, x_, y_, thickness_data_, sst_data_ = \
        data_loader.load_raw_data()
    
    # Preprocess (includes thinning, leap day removal, bbox cropping)
    data, data_mean_month, data_mean_week, x, y, mask_ice, mask_land, ny, nx, thickness_data, sst_data, bbox = \
        data_loader.preprocess_data(mask_land_, mask_ice_, data_, data_mean_month_, data_mean_week_, x_, y_, thickness_data_, sst_data_)
    
    # Split data (selecting years 1993-2023 by slicing [4:-1])
    data = data[4:-1]
    thickness_data = thickness_data[:len(data)]  # Match data length
    sst_data = sst_data[:len(data)]  # Match data length
    
    print(f"\n✓ Data loaded and preprocessed")
    print(f"   Years loaded: {len(data)}")
    print(f"   Grid shape: ({ny}, {nx})")
    print(f"   Bounding box: {bbox}")
    print(f"   Thin factor: {config.thin}")
    
    # Define region mask (active ice region)
    region_mask = mask_ice.astype(bool)
    n_pixel_region = region_mask.sum()
    print(f"   Active pixels in region: {n_pixel_region}")
    
    # ========================================================================
    # DATA SPLIT - Use exact same split as training  
    # ========================================================================
    print("\n" + "-"*80)
    print("DATA SPLIT CONFIGURATION")
    print("-"*80)
    
    # Split data (same as main script)
    data_train, data_val, data_test, data_tot, n_year_train, n_year_val, n_year_test, ny, nx = \
        data_loader.split_train_val_test(data)
    
    n_year_tot = n_year_train + n_year_val + n_year_test
    
    print(f"   Train: {n_year_train} years")
    print(f"   Val:   {n_year_val} years")
    print(f"   Test:  {n_year_test} years")
    print(f"   Total: {n_year_tot} years")
    
    # ========================================================================
    # LOAD DMD FORECASTS - Using DMDBaseline class
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: LOAD DMD BASELINE")
    print("="*80)
    
    dmd_baseline = DMDBaseline(config, bbox=bbox)
    dmd_years, y_dmd_pred, y_dmd_std = dmd_baseline.load_dmd_predictions()
    
    if y_dmd_pred is None:
        raise RuntimeError("❌ CRITICAL: DMD predictions could not be loaded")
    
    print(f"✓ DMD forecasts loaded and cropped")
    print(f"   Shape: {y_dmd_pred.shape}")
    print(f"   Years: {dmd_years[0]} to {dmd_years[-1]} ({len(dmd_years)} years)")
    
    # ========================================================================
    # PREPARE LOW-FIDELITY DATA (x1, x2, x3 levels)
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: LOW-FIDELITY DATA PREPARATION")
    print("="*80)
    
    # Initialize with correct constructor parameters
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
    
    # Apply POD reduction
    x1_train, x1_val, x1_test, x2_train, x2_val, x2_test, pod_data = \
        lf_prep.apply_pod_reduction(
            x1_train, x1_val, x1_test,
            x2_train, x2_val, x2_test,
            threshold=0.9  # Use modes that explain 90% of variance
        )
    
    # Level 3: Sensors (DMD residuals sampled at random locations)
    # NOTE: Use n_sensors=80 to match the trained model checkpoint
    clim_data_sensors, sensor_mask, sensor_idxs, n_sensors = lf_prep.prepare_sensors(
        y_true_data=data,
        y_dmd_pred=y_dmd_pred,
        dmd_years=dmd_years,
        n_sensors=80,  # CRITICAL: Must match checkpoint (80, not 128)
        seed=0,
        sensor_noise_std=0.02  # Add 2% noise to simulate sensor uncertainty
    )
    
    # Split sensor data
    x3_train, x3_val, x3_test = lf_prep.split_sensor_data(
        clim_data_sensors, n_year_train, n_year_val, n_year_test
    )
    
    # ========================================================================
    # PREPARE HIGH-FIDELITY TARGET (y)
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
    
    # Store residual scaling parameters
    residual_scaler = {
        'mean': train_mean,
        'std': train_std
    }
    
    # ========================================================================
    # DATA SCALING
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
    # PYTORCH DATASET CREATION
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
    
    # ========================================================================
    # PREPARE GROUND TRUTH AND BASELINES FOR CALIBRATION/EVALUATION
    # ========================================================================
    print("\n" + "-"*80)
    print("PREPARING GROUND TRUTH AND BASELINES FOR CALIBRATION/EVALUATION")
    print("-"*80)
    
    # For calibration, we need:
    # 1. DMD forecasts (baseline) on ice mask pixels
    # 2. Original TRUE SIC data on ice mask pixels
    
    # Extract DMD validation years
    dmd_val_years = y_dmd_pred[n_year_train:n_year_train + n_year_val]
    dmd_val_continuous = dmd_val_years.reshape(-1, ny, nx)
    val_baseline_active = dmd_val_continuous[:, region_mask]
    val_baseline_active_tensor = torch.tensor(val_baseline_active, dtype=torch.float32).to(config.device)
    
    print(f"✓ DMD validation baseline: {val_baseline_active_tensor.shape}")
    
    # Extract validation years from original data
    y_true_val_years = data[n_year_train:n_year_train + n_year_val]
    y_true_val_stacked = np.stack(y_true_val_years, axis=0)  # (n_val_years, 365, ny, nx)
    y_true_val_continuous = y_true_val_stacked.reshape(-1, ny, nx)
    y_true_val_active = y_true_val_continuous[:, region_mask]
    y_true_val_tensor = torch.tensor(y_true_val_active, dtype=torch.float32).to(config.device)
    
    print(f"✓ True SIC validation: {y_true_val_tensor.shape}")
    
    # Extract DMD test years
    dmd_test_years = y_dmd_pred[n_year_train + n_year_val:n_year_train + n_year_val + n_year_test]
    dmd_test_continuous = dmd_test_years.reshape(-1, ny, nx)
    test_baseline_active = dmd_test_continuous[:, region_mask]
    test_baseline_active_tensor = torch.tensor(test_baseline_active, dtype=torch.float32).to(config.device)
    
    print(f"✓ DMD test baseline: {test_baseline_active_tensor.shape}")
    
    # Extract test years from original data
    y_true_test_years = data[n_year_train + n_year_val:n_year_train + n_year_val + n_year_test]
    y_true_test_stacked = np.stack(y_true_test_years, axis=0)
    y_true_test_continuous = y_true_test_stacked.reshape(-1, ny, nx)
    y_true_test_active = y_true_test_continuous[:, region_mask]
    y_true_test_tensor = torch.tensor(y_true_test_active, dtype=torch.float32).to(config.device)
    
    print(f"✓ True SIC test: {y_true_test_tensor.shape}")
    
    # ========================================================================
    # CREATE MODEL - Using ModelSetup exactly as main script
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 7: CREATE AND LOAD MODEL")
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
    
    # Use ModelSetup to create model (exact same as training)
    model_setup = ModelSetup(config, levels_dim, n_pixel_region, train_dataset)
    model = model_setup.create_model()
    
    # Load checkpoint
    print(f"\n📥 Loading model weights from checkpoint...")
    state_dict = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # ========================================================================
    # CONFORMAL CALIBRATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 8: CONFORMAL CALIBRATION")
    print("="*80)
    
    # Create a smaller batch dataloader for calibration to reduce GPU memory pressure
    calibration_batch_size = max(1, val_loader.batch_size // 4)  # Use 1/4 of training batch size
    cal_loader = DataLoader(
        val_loader.dataset,
        batch_size=calibration_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print(f"Using calibration batch size: {calibration_batch_size} (reduced from {val_loader.batch_size} for memory safety)")
    
    calibrator = ConformalCalibration(
        config, model, 
        pixelwise=True, 
        temporal=True, 
        n_seasons=4
    )
    
    q_scores = calibrator.calibrate_conditional(
        cal_loader, 
        residual_scaler, 
        val_baseline_active_tensor, 
        y_true_val_tensor
    )
    
    print(f"✓ Calibration completed")
    print(f"   Scenarios calibrated: {list(q_scores.keys())}")
    for scenario_name, q_score in q_scores.items():
        if hasattr(q_score, 'shape'):
            print(f"     {scenario_name}: {q_score.shape}")
    
    # ========================================================================
    # GENERATE PREDICTIONS FOR ALL SCENARIOS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 9: GENERATE PREDICTIONS FOR ALL SCENARIOS")
    print("="*80)
    
    evaluator = TestEvaluator(
        config, model, q_scores, 
        calibrator.scenarios, residual_scaler
    )
    
    scenario_predictions = {}
    
    for scenario_name, mask_cfg in evaluator.scenarios.items():
        print(f"\n  🔄 Processing: {scenario_name}")
        print(f"     Mask config: {mask_cfg}")
        
        vec_low, vec_med, vec_high, vec_true = evaluator.generate_full_sic_predictions(
            test_loader, 
            test_baseline_active_tensor, 
            q_scores[scenario_name],
            y_true_test_tensor, 
            mask_config=mask_cfg
        )
        
        scenario_predictions[scenario_name] = (vec_low, vec_med, vec_high, vec_true)
        
        print(f"     ✓ Complete - Shape: {vec_med.shape}")
    
    print("\n" + "="*80)
    print("✅ ALL PREDICTIONS GENERATED")
    print("="*80)
    
    # ========================================================================
    # SAVE PREDICTIONS
    # ========================================================================
    print("\n💾 Saving predictions for all scenarios...")
    
    # Build predictions dict with all scenarios
    predictions_dict = {}
    
    for scenario_name, (vec_low, vec_med, vec_high, vec_true) in scenario_predictions.items():
        # CRITICAL FIX: Reshape from (B, S, n_pixels) to (B*S, n_pixels) = (time, pixels)
        # The generate_full_sic_predictions returns 3D tensors with a batch dimension
        # We need 2D arrays for metrics: (timesteps, pixels)
        n_pixels = vec_low.shape[-1]
        
        vec_low_2d = torch.clamp(vec_low, 0, 1).view(-1, n_pixels)
        vec_med_2d = torch.clamp(vec_med, 0, 1).view(-1, n_pixels)
        vec_high_2d = torch.clamp(vec_high, 0, 1).view(-1, n_pixels)
        vec_true_2d = torch.clamp(vec_true, 0, 1).view(-1, n_pixels)
        
        predictions_dict[scenario_name] = {
            'low': vec_low_2d.cpu().numpy(),
            'median': vec_med_2d.cpu().numpy(),
            'high': vec_high_2d.cpu().numpy(),
            'true': vec_true_2d.cpu().numpy()
        }
        print(f"  ✓ {scenario_name}: shape {predictions_dict[scenario_name]['median'].shape} (time, pixels)")
    
    # Create final data structure
    data_dict = {
        'q_scores': q_scores,
        'predictions': predictions_dict
    }
    
    save_path = config.results_dir / "all_predictions.pkl"
    
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"\n✓ All numerical data saved to: {save_path}")
    print(f"  Scenarios saved: {list(predictions_dict.keys())}")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\n📦 All predictions saved to:")
    print(f"   {save_path}")
    print(f"\n📊 Scenarios saved: {list(scenario_predictions.keys())}")
    print(f"\n📝 Next steps:")
    print(f"   1. Open IceNet_Metrics_Analysis.ipynb")
    print(f"   2. Run all cells to compute metrics for all scenarios")
    print(f"   3. Each scenario will now have distinct predictions and metrics")
    print("="*80)


if __name__ == "__main__":
    main()
