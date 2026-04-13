"""Orchestration workflow for the multifidelity pipeline."""

from COMETQR_shared import *
from COMETQR_config_data import Config, DataLoader_Ice, ClimatologyBaseline, DMDBaseline
from COMETQR_preprocessing import LowFidelityDataPrep, HighFidelityTargetPrep, DataScaler, TorchDatasetPreparation
from COMETQR_training import ModelSetup, ModelTrainer
from COMETQR_evaluation import ConformalCalibration, TestEvaluator
from COMETQR_results import ResultsSaver

def main():
    """Main execution pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multifidelity Transformer GPU Execution")
    
    parser.add_argument("--base_path", type=str, 
                       default="/work/u10715220",  # Default su WORK per i dati
                       help="Path to DATA and RESULTS (WORK)")
                       
    parser.add_argument("--project_path", type=str, 
                       default=".",  # Default alla cartella corrente (HOME) per il codice
                       help="Path to code and configuration files")

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
    
    # PHASE 1: CONFIGURATION AND DATA LOADING
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
    
    # PHASE 2: BASELINE COMPUTATIONS

    # PHASE 3: LOAD PRE-COMPUTED DMD FORECASTS
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
    
    # LOAD PRE-COMPUTED DMD FORECASTS (WITH SPATIAL CROPPING)
    print("\n🔍 LOADING PRE-COMPUTED DMD FORECASTS...")
    
    dmd_baseline = DMDBaseline(config, bbox=bbox)
    dmd_years, y_dmd_pred, y_dmd_std = dmd_baseline.load_dmd_predictions()
    
    if y_dmd_pred is None:
        raise RuntimeError("❌ CRITICAL: DMD predictions could not be loaded. Cannot proceed.")
    
    print(f"✓ DMD forecasts loaded and cropped to match data domain")
    print(f"  Shape: {y_dmd_pred.shape}")
    print(f"  Years: {dmd_years[0]} to {dmd_years[-1]} ({len(dmd_years)} years)")

    # PHASE 3: LOW-FIDELITY DATA PREPARATION
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
    
    # PHASE 4: HIGH-FIDELITY TARGET PREPARATION
    # CRITICAL WORKFLOW CONSISTENCY:
    # - Residual statistics (mean/std) computed ONLY on ice mask pixels
    # - Training loss computed ONLY on ice mask pixels
    # - Conformal calibration computed ONLY on ice mask pixels
    # This ensures all components use the same pixel distribution for coherency
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
    
    # PHASE 5: DATA SCALING
    print("\n" + "="*80)
    print("PHASE 5: DATA SCALING")
    print("="*80)
    
    scaler = DataScaler(config)
    x1_train, x1_val, x1_test, x2_train, x2_val, x2_test, x3_train, x3_val, x3_test, scalers = \
        scaler.scale_all_levels(x1_train, x1_val, x1_test,
                               x2_train, x2_val, x2_test,
                               x3_train, x3_val, x3_test)
    
    # PHASE 6: PYTORCH DATASET CREATION
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
    

    # MEMORY CLEANUP (CRITICO PER EVITARE OOM)
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

    
    # PHASE 7: MODEL SETUP
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
    
    # PHASE 8: MODEL TRAINING
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
    
    # PHASE 9: CONFORMAL CALIBRATION
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
    
    # LOAD DMD FORECASTS (REQUIRED FOR VALIDATION DATA PREPARATION)
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
    
    # PREPARE DMD AND TRUE SIC FOR VALIDATION (NEEDED FOR CALIBRATION)
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
    
    # PHASE 10: TEST EVALUATION
    print("\n" + "="*80)
    print("PHASE 10: TEST EVALUATION")
    print("="*80)
    
    evaluator = TestEvaluator(config, model, q_scores, calibrator.scenarios, residual_scaler)
    
    # PREPARE DMD TEST DATA
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
    
    # LOAD ORIGINAL TRUE SIC DATA FOR GROUND TRUTH (BEFORE EVALUATION!)
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
    
    # NOW EVALUATE WITH ACTUAL GROUND TRUTH SIC
    print("\n" + "="*70)
    print("EVALUATING MODEL WITH ORIGINAL GROUND TRUTH SIC")
    print("="*70)
    
    # Evaluate - pass actual ground truth SIC
    df_results = evaluator.evaluate_test_physics(test_loader, dmd_test_flat, region_mask_dmd, y_true_test_tensor)
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(df_results.to_string(index=False))
    
    # Ground truth already loaded above, proceed with other results
    # PHASE 10B: GENERATE PREDICTIONS FOR ALL SCENARIOS
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
    
    # PHASE 10C: COMPUTE CLIMATOLOGY FROM TRAIN + VAL DATA
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
    
    # PHASE 11: SAVE RESULTS
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
    
    # COMPLETION
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
