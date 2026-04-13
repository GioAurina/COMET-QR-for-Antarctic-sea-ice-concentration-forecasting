"""Metrics and result-saving components for the multifidelity pipeline."""

from COMETQR_shared import *

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




