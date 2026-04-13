"""Calibration and evaluation components for the multifidelity pipeline."""

from COMETQR_shared import *

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




# ICENET-STYLE METRICS FOR COMPARISON

