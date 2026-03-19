# IceNet-Style Analysis Additions
# To be integrated into the notebook

"""
Key IceNet Analyses (from Nature Communications 2021):

1. Sea Ice Extent (SIE) Analysis
   - Compare predicted vs observed total ice extent
   - Monthly time series
   - Error bars from uncertainty quantiles

2. Spatial Binary Accuracy Maps
   - Per-pixel accuracy over test period
   - Highlight regions of high/low performance

3. Monthly Performance (not just seasonal)
   - IIEE and accuracy by month
   - Identify seasonal trends

4. Regional Analysis
   - Break down by Antarctic sectors (if applicable)
   - Weddell Sea, Ross Sea, etc.

5. SIE Error Metrics
   - Mean Absolute Error in SIE
   - Normalized SIE Error

6. Climatology Baseline Comparison
   - Show improvement over simple climatology

7. Lead Time Analysis (if multi-month forecasts available)
   - Performance degradation with forecast horizon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def calculate_sie(sic_data, ice_threshold=0.15, grid_area_km2=625):
    """
    Calculate Sea Ice Extent (SIE) from SIC data.
    
    SIE = total area where SIC > threshold
    
    Args:
        sic_data: (time, pixels) array of SIC values [0, 1]
        ice_threshold: Threshold for ice presence (default: 0.15)
        grid_area_km2: Area of each grid cell in km²
        
    Returns:
        sie: (time,) array of SIE in km²
    """
    ice_mask = sic_data > ice_threshold
    sie = np.sum(ice_mask, axis=1) * grid_area_km2
    return sie


def calculate_spatial_binary_accuracy(pred_sic, true_sic, ice_threshold=0.15):
    """
    Calculate per-pixel binary accuracy over time.
    
    Args:
        pred_sic: (time, pixels) predictions
        true_sic: (time, pixels) ground truth
        ice_threshold: Ice edge threshold
        
    Returns:
        pixel_accuracy: (pixels,) accuracy for each pixel
    """
    pred_ice = (pred_sic > ice_threshold).astype(float)
    true_ice = (true_sic > ice_threshold).astype(float)
    
    # Count correct predictions for each pixel across time
    correct = (pred_ice == true_ice).astype(float)
    pixel_accuracy = np.mean(correct, axis=0)
    
    return pixel_accuracy


def calculate_monthly_metrics(pred_sic, true_sic, days_per_year=365, grid_area=625, ice_threshold=0.15):
    """
    Calculate IIEE and accuracy by month.
    
    Args:
        pred_sic: (time, pixels) predictions
        true_sic: (time, pixels) ground truth
        days_per_year: Days per year (365 for no leap days)
        grid_area: Grid cell area in km²
        ice_threshold: Ice edge threshold
        
    Returns:
        monthly_metrics: DataFrame with metrics per month
    """
    n_timesteps = pred_sic.shape[0]
    n_pixels = pred_sic.shape[1]
    
    # Approximate days per month
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cumulative_days = np.cumsum([0] + days_per_month)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    results = []
    
    for year_start in range(0, n_timesteps, days_per_year):
        year_idx = year_start // days_per_year
        
        for month_idx in range(12):
            month_start = year_start + cumulative_days[month_idx]
            month_end = year_start + cumulative_days[month_idx + 1]
            
            if month_end <= n_timesteps:
                # Extract month data
                pred_month = pred_sic[month_start:month_end]
                true_month = true_sic[month_start:month_end]
                
                # Calculate IIEE
                pred_ice = (pred_month > ice_threshold).astype(float)
                true_ice = (true_month > ice_threshold).astype(float)
                
                O_mask = (pred_ice > true_ice)
                U_mask = (true_ice > pred_ice)
                
                iiee_per_day = (np.sum(O_mask, axis=1) + np.sum(U_mask, axis=1)).mean()
                iiee = iiee_per_day * grid_area
                
                # Binary accuracy
                total_area = n_pixels * grid_area
                binary_acc = (1 - iiee / total_area) * 100
                
                results.append({
                    'year': year_idx,
                    'month': month_names[month_idx],
                    'month_num': month_idx + 1,
                    'iiee_km2': iiee,
                    'binary_accuracy': binary_acc
                })
    
    return pd.DataFrame(results)


def plot_sie_comparison(pred_sic, true_sic, dates=None, ice_threshold=0.15, grid_area=625, 
                        scenario_name='Model', save_path=None):
    """
    Plot SIE comparison (IceNet-style Figure 2).
    
    Args:
        pred_sic: (time, pixels) predictions
        true_sic: (time, pixels) ground truth
        dates: Optional date labels
        ice_threshold: Ice threshold
        grid_area: Grid cell area
        scenario_name: Model name for legend
        save_path: Path to save figure
    """
    # Calculate SIE
    sie_pred = calculate_sie(pred_sic, ice_threshold, grid_area) / 1e6  # Convert to million km²
    sie_true = calculate_sie(true_sic, ice_threshold, grid_area) / 1e6
    
    # Create time axis (days)
    time = np.arange(len(sie_pred))
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Top: SIE Time Series
    ax1 = axes[0]
    ax1.plot(time, sie_true, label='Observed', color='black', linewidth=1.5, alpha=0.8)
    ax1.plot(time, sie_pred, label=scenario_name, color='steelblue', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Sea Ice Extent (10⁶ km²)', fontsize=12, fontweight='bold')
    ax1.set_title('Sea Ice Extent Comparison (IceNet-Style)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: SIE Error
    ax2 = axes[1]
    sie_error = sie_pred - sie_true
    ax2.plot(time, sie_error, color='coral', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(time, 0, sie_error, alpha=0.3, color='coral')
    ax2.set_xlabel('Days', fontsize=12, fontweight='bold')
    ax2.set_ylabel('SIE Error (10⁶ km²)', fontsize=12, fontweight='bold')
    ax2.set_title('Forecast Error', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # Stats
    mae = np.mean(np.abs(sie_error))
    rmse = np.sqrt(np.mean(sie_error**2))
    
    ax2.text(0.02, 0.98, f'MAE: {mae:.3f} × 10⁶ km²\nRMSE: {rmse:.3f} × 10⁶ km²',
             transform=ax2.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved SIE comparison to {save_path}")
    
    return fig


def plot_spatial_accuracy_map(pixel_accuracy, region_mask, mask_land, x, y, 
                               scenario_name='Model', save_path=None):
    """
    Plot spatial binary accuracy map (IceNet-style Figure 3).
    
    Args:
        pixel_accuracy: (pixels,) accuracy per pixel
        region_mask: Boolean mask for active region
        mask_land: Land mask  
        x, y: Coordinate arrays
        scenario_name: Model name
        save_path: Save path
    """
    ny, nx = mask_land.shape
    
    # Reconstruct 2D accuracy map
    accuracy_map = np.full((ny, nx), np.nan)
    accuracy_map[region_mask] = pixel_accuracy * 100  # Convert to percentage
    
    # Mask land
    accuracy_map_masked = np.ma.masked_where(mask_land, accuracy_map)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.pcolormesh(x, y, accuracy_map_masked, 
                       cmap='RdYlGn', vmin=80, vmax=100, shading='auto')
    
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
    cbar.set_label('Binary Accuracy (%)', fontsize=12, fontweight='bold')
    
    ax.set_title(f'Spatial Binary Accuracy - {scenario_name}\n(IceNet-Style Analysis)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved spatial accuracy map to {save_path}")
    
    return fig
