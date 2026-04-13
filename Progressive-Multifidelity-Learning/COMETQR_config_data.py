"""Configuration and data-loading components for the multifidelity pipeline."""

from COMETQR_shared import *

class Config:
    """Configuration container for all experiment settings"""
    
    def __init__(self, args):
        self.args = args
        self.setup_paths()
        self.load_experiment_config()
        # 3. Setup Device
        self.setup_device()
        # 4. Set thin parameter (spatial downsampling factor)
        self.thin = 2  # 432x432 -> 216x216 grid
        
    def setup_paths(self):
        """Setup all directory paths"""
        self.base_path = Path(self.args.base_path)

        if hasattr(self.args, 'project_path') and self.args.project_path:
            self.project_path = Path(self.args.project_path)
        else:
            self.project_path = Path(".").resolve()

        if hasattr(self.args, 'scratch_path') and self.args.scratch_path:
            self.scratch_path = Path(self.args.scratch_path)
        else:
            self.scratch_path = Path("/scratch_global/u10715220")

        # Se specifichiamo --output_dir, usiamo quello. Altrimenti usiamo base_path (Work).
        if self.args.output_dir:
            self.output_root = Path(self.args.output_dir)
            print(f"🚀 OUTPUTS REDIRECTED TO: {self.output_root}")
        else:
            self.output_root = self.base_path

        self.data_path = self.base_path / "data" / "ice"

        self.checkpoint_dir = self.output_root / "checkpoints"
        self.results_dir = self.output_root / "Results"
        self.scaler_dir = self.output_root / "scalers"

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
        
        config_path = self.project_path / "multifidelity_transformer" / "experiment_configurations" / experiment_name / f"{experiment_number}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"❌ ERRORE CRITICO: Config non trovato in: {config_path}")

        print(f"📖 Loading config from: {config_path}")
        with open(config_path, 'r') as file:
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

# DATA LOADING AND PREPROCESSING

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


