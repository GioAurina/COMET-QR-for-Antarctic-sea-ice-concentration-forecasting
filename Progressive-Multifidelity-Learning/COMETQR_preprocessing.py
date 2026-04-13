"""Preprocessing and dataset-preparation components for the multifidelity pipeline."""

from COMETQR_shared import *

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
        
        # POD for Level 1 (Ice Thickness)
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
        # x1_train_reduced = x1_train_reduced / S1[:n_POD_x1]
        # x1_val_reduced = x1_val_reduced / S1[:n_POD_x1]
        # x1_test_reduced = x1_test_reduced / S1[:n_POD_x1]
        
        print(f"✓ Reduced shapes: train{x1_train_reduced.shape}, val{x1_val_reduced.shape}, test{x1_test_reduced.shape}")
        
        # POD for Level 2 (SST)
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
        # x2_train_reduced = x2_train_reduced / S2[:n_POD_x2]
        # x2_val_reduced = x2_val_reduced / S2[:n_POD_x2]
        # x2_test_reduced = x2_test_reduced / S2[:n_POD_x2]
        
        print(f"✓ Reduced shapes: train{x2_train_reduced.shape}, val{x2_val_reduced.shape}, test{x2_test_reduced.shape}")
        
        # Summary
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
        if len(self.data) == len(self.dmd_years):
            print(f"DEBUG: Data length ({len(self.data)}) matches DMD years. Using direct alignment.")
            y_true = np.array(self.data)
        else:
            print(f"DEBUG: Data length ({len(self.data)}) != DMD years ({len(self.dmd_years)}). Using indexing.")
            y_true = np.array([self.data[year] for year in self.dmd_years])
            
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


