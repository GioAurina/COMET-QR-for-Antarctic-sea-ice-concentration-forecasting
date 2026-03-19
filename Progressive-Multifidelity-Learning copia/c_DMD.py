"""
DMD Analysis Pipeline - GPU Optimized Version
Converts notebook to production script with automated result saving
"""

import os
import pickle
import dill
from itertools import accumulate
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt

from pydmd import DMD, BOPDMD, FbDMD, MrDMD
from pydmd.plotter import plot_eigs, plot_summary, plot_modes_2D
from pydmd.preprocessing import hankel_preprocessing
from scipy.ndimage import gaussian_filter
from tqdm.autonotebook import tqdm, trange
from scipy.signal import fftconvolve

import sys
sys.path.append('./src/modules/')

from plot_jupyter import contour_compare, contour_data
from data_wrangle import get_days_before, get_test_set, window_mean, thin_data, del_leap
from dmd_routines import reshape_data2dmd, train_dmd, reshape_Psi2data, eval_dmd, eval_dmd_latent, bootstrap_train_dmd, eval_dmd_ensemble, eval_dmd_latent

import xarray as xr
from dataclasses import dataclass

# Parameters
year = 32
day = 0
window = 1  # Temporal smoothing window (1 = disabled, avoids alignment issues)
thin = 2
rank = 5
N_boot_strap = 100
eig_constraints = {"stable", "conjugate_pairs"}
threshold = 0.3
T_pred = 730
T_train = 365 * 2

# ============================================================================
# CONFIGURATION
# ============================================================================

WORK_DIR = '/work/u10715220'

# --- MODIFICA SALVA-SPAZIO ---
# Salviamo gli output su Scratch Global per evitare "Disk Full"
SCRATCH_DIR = '/scratch_global/u10715220'

# Crea le cartelle se non esistono
if not os.path.exists(SCRATCH_DIR):
    os.makedirs(SCRATCH_DIR)

RESULTS_DIR = os.path.join(SCRATCH_DIR, 'Results')
CHECKPOINTS_DIR = os.path.join(SCRATCH_DIR, 'checkpoints')
# -----------------------------

# Assicuriamo che Python crei queste cartelle se non esistono
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# 3. INPUT DATI
# Assumiamo che il file .pkl sia direttamente nella radice di u10715220
# Se è in una sottocartella, modifica la riga sotto (es: os.path.join(WORK_DIR, 'data', 'nomefile.pkl'))
DATA_FILE = os.path.join(WORK_DIR, 'data/ice/Antarctic_years_1989_2024i.pkl')

print("="*80)
print(f"DMD PIPELINE CONFIGURATION")
print(f"[-] Work Directory: {WORK_DIR}")
print(f"[-] Checkpoints will be saved to: {CHECKPOINTS_DIR}") # <--- Verifica questo print nel log
print(f"[-] Results will be saved to:     {RESULTS_DIR}")
print(f"[-] Input Data file:              {DATA_FILE}")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"ERRORE CRITICO: Il file dati non esiste in: {DATA_FILE}")

with open(DATA_FILE, 'rb') as f:
    mask_land_, mask_ice_, data_, data_mean_month_, data_mean_week_, x_, y_ = dill.load(f)

data, data_mean_month, data_mean_week, x, y, mask_ice, mask_land = \
    thin_data(thin, data_, data_mean_month_, data_mean_week_, x_, y_, mask_ice_, mask_land_)

data = del_leap(data)
del data_, data_mean_month_, data_mean_week_
data = data[2:-1]  # Use years 2 to 34 (1991-2023)
ny, nx = data[0].shape[1:]

years = np.arange(2, 33)
X0_sample = get_days_before(data, years[0], day, T_train + window - 1)
X0_sample_wm = window_mean(X0_sample, window)
actual_time_steps = X0_sample_wm.shape[0]  # Calculate AFTER window_mean

previous_days = np.zeros((len(years), X0_sample.shape[0], ny, nx), dtype=np.float32)
previous_wmeans = np.zeros((len(years), actual_time_steps, ny, nx), dtype=np.float32)

for idx, year in enumerate(tqdm(years, desc="Loading years")):
    X0_ = get_days_before(data, year, day, T_train + window - 1)
    previous_days[idx] = X0_
    X0_wm = window_mean(X0_, window)
    previous_wmeans[idx] = X0_wm

t_train = np.arange(-T_train, 0)
print(f"Data loaded: {len(years)} years, shape per year: {previous_wmeans[0].shape}\n")

# ============================================================================
# PREPARE DMD SNAPSHOTS
# ============================================================================
print("Preparing DMD snapshots with time delay...")
X_delayed = []
for year in years:
    X0_ = previous_wmeans[year - 2]
    X_d, t_delayed, data_shape = reshape_data2dmd(X0_, t_train, time_delay=2, 
                                            mask=mask_ice, isKeepFirstTimes=True)
    X_delayed.append(X_d)

X_delayed = np.array(X_delayed)
X_delayed_reg = X_delayed
print(f"DMD snapshots prepared: shape {X_delayed.shape}\n")

# ============================================================================
# TRAIN DMD WITH BOOTSTRAP
# ============================================================================
print(f"Training DMD for {len(years)} years...")

# MODIFICA: Inizializziamo con None per mantenere la lunghezza fissa (es. 31)
L_tot = [None] * len(years)
Psi_tot = [None] * len(years)
bn_tot = [None] * len(years)
failed_years = []

for year_idx, X_d in enumerate(tqdm(X_delayed_reg, desc="Training DMD")):
    year = years[year_idx]
    
    try:
        L_s, Psi_s_, bn_s = bootstrap_train_dmd(
            N_boot_strap, X_d, t_delayed, svd_rank=rank, eig_constraints=eig_constraints
        )
        
        Psi_s = np.zeros((N_boot_strap, rank, ny, nx), dtype=complex)
        for i, Psi_ in enumerate(Psi_s_):
            Psi_s[i] = reshape_Psi2data(Psi_, data_shape, mask=mask_ice)
        
        # MODIFICA: Assegnazione per indice (mantiene il posto anche se altri anni falliscono)
        L_tot[year_idx] = L_s
        Psi_tot[year_idx] = Psi_s
        bn_tot[year_idx] = bn_s
        
    except Exception as e: # Cattura qualsiasi errore (SVD, Memory, ecc.)
        print(f"DMD failed for year {year}: {e}")
        failed_years.append(year_idx)

print(f"Training complete: {len(years) - len(failed_years)}/{len(years)} successful\n")

# ============================================================================
# OUTLIER FILTERING
# ============================================================================
print("Computing outlier filters...")
threshold_outlier = threshold
ns_all_years = []
inds_good_all_years = []

for year_idx in tqdm(range(len(years)), desc="Filtering outliers"):
    # MODIFICA: Se l'anno non è stato calcolato, riempiamo con placeholder e saltiamo
    if L_tot[year_idx] is None:
        ns_all_years.append(np.full(N_boot_strap, 9999.9)) # Valore alto fittizio
        inds_good_all_years.append(np.zeros(N_boot_strap, dtype=bool)) # Nessun "good"
        continue

    year = years[year_idx]
    X0_year = previous_wmeans[year_idx]
    # ... (il resto del codice nel ciclo rimane identico) ...
    X0_delayed = X0_year[:len(t_delayed)]
    
    L_s = L_tot[year_idx]
    Psi_s = Psi_tot[year_idx]
    bn_s = bn_tot[year_idx]
    
    X_train_ensemble = eval_dmd_ensemble(L_s, Psi_s, bn_s, t_delayed, isPositive=True)
    
    X0_flat = X0_delayed.reshape(len(t_delayed), -1)
    X0_norm = np.linalg.norm(X0_flat)
    
    ns = np.zeros(N_boot_strap)
    for i in range(N_boot_strap):
        X_train_flat = X_train_ensemble[i].reshape(len(t_delayed), -1)
        ns[i] = np.linalg.norm(X_train_flat - X0_flat) / X0_norm
    
    inds_good = ns < threshold_outlier
    ns_all_years.append(ns)
    inds_good_all_years.append(inds_good)
    del X_train_ensemble

# ... (il resto fuori dal ciclo rimane uguale)

ns_all_years = np.array(ns_all_years)
inds_good_all_years = np.array(inds_good_all_years)

n_good_per_year = inds_good_all_years.sum(axis=1)
n_outliers_per_year = N_boot_strap - n_good_per_year

print(f"Outlier filtering complete: avg {n_good_per_year.mean():.1f}/{N_boot_strap} good realizations per year\n")

# Save outlier statistics plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1 = axes[0]
ax1.bar(years, n_good_per_year, color='green', alpha=0.6, label='Good realizations')
ax1.bar(years, n_outliers_per_year, bottom=n_good_per_year, color='red', alpha=0.6, label='Outliers')
ax1.set_xlabel('Year Index')
ax1.set_ylabel('Number of Bootstrap Realizations')
ax1.set_title('Bootstrap Realizations: Good vs Outliers by Year')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.boxplot(ns_all_years.T, positions=years, widths=0.6)
ax2.axhline(threshold_outlier, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold_outlier}')
ax2.set_xlabel('Year Index')
ax2.set_ylabel('Normalized Distance')
ax2.set_title('Distribution of Normalized Distances by Year')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'outlier_statistics.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# APPLY FILTERS AND SAVE FILTERED DMD
# ============================================================================
print("Applying filters to DMD coefficients...")

# MODIFICA: List comprehension sicura
L_tot_filtered = [L_tot[i][inds_good_all_years[i]] if L_tot[i] is not None else None for i in range(len(years))]
Psi_tot_filtered = [Psi_tot[i][inds_good_all_years[i]] if Psi_tot[i] is not None else None for i in range(len(years))]
bn_tot_filtered = [bn_tot[i][inds_good_all_years[i]] if bn_tot[i] is not None else None for i in range(len(years))]

# ... (il salvataggio pickle rimane uguale) ...

save_path_filtered = os.path.join(CHECKPOINTS_DIR, f'dmd_fits_FILTERED_rank{rank}_bootstrap{N_boot_strap}_thin{thin}_window{window}_years2-34.pkl')

dmd_filtered_results = {
    'L_tot': L_tot_filtered,
    'Psi_tot': Psi_tot_filtered,
    'bn_tot': bn_tot_filtered,
    'inds_good_all_years': inds_good_all_years,
    'years': years,
    'data_shape': data_shape,
    'metadata': {
        'rank': rank,
        'N_boot_strap_original': N_boot_strap,
        'N_boot_strap_filtered': inds_good_all_years.sum(axis=1).tolist(),
        'threshold': threshold_outlier,
        'T_train': T_train,
        'window': window,
        'day': day,
        'thin': thin,
        'nx': nx,
        'ny': ny,
        'eig_constraints': eig_constraints,
    }
}

with open(save_path_filtered, 'wb') as f:
    dill.dump(dmd_filtered_results, f)

L_tot = L_tot_filtered
Psi_tot = Psi_tot_filtered
bn_tot = bn_tot_filtered

print(f"Filtered DMD saved: {save_path_filtered}\n")

# ============================================================================
# GENERATE FORECASTS
# ============================================================================
print("Generating forecasts for all years...")
t_test = np.arange(0, 365)

I_pred_mean_all = []
I_pred_std_all = []
I_true_all = []
z_pred_mean_all = []
z_pred_std_all = []
y_pred_mean_all = []
y_pred_std_all = []

for year_idx in tqdm(range(len(years)), desc="Forecasting"):
    # MODIFICA: Salta gli anni falliti
    if L_tot[year_idx] is None:
        # Aggiungi placeholder vuoti per mantenere l'allineamento dei grafici (opzionale ma consigliato)
        nan_array_scalar = np.full(365, np.nan)
        nan_array_spatial = np.full((365, ny, nx), np.nan)
        
        I_pred_mean_all.append(nan_array_scalar)
        I_pred_std_all.append(nan_array_scalar)
        I_true_all.append(data[years[year_idx]] if year_idx < len(data) else nan_array_spatial) # Fallback
        
        z_pred_mean_all.append(np.full((365, rank), np.nan))
        z_pred_std_all.append(np.full((365, rank), np.nan))
        y_pred_mean_all.append(nan_array_spatial)
        y_pred_std_all.append(nan_array_spatial)
        continue
    year = years[year_idx]
    
    L_s = L_tot[year_idx]
    Psi_s = Psi_tot[year_idx]
    bn_s = bn_tot[year_idx]
    n_good_year = L_s.shape[0]
    
    y_pred_ensemble = eval_dmd_ensemble(L_s, Psi_s, bn_s, t_test, isPositive=True)
    
    z_latent_ensemble = np.zeros((n_good_year, 365, rank), dtype=complex)
    for i in range(n_good_year):
        z_latent_ensemble[i] = eval_dmd_latent(L_s[i], bn_s[i], t_test, isPositive=False)
    
    I_pred_ensemble = np.trapezoid(np.trapezoid(y_pred_ensemble, y, axis=3), x, axis=2)
    y_true = data[year]
    I_true = np.trapezoid(np.trapezoid(y_true, y, axis=2), x, axis=1)
    
    I_pred_mean_all.append(I_pred_ensemble.mean(axis=0))
    I_pred_std_all.append(I_pred_ensemble.std(axis=0))
    I_true_all.append(I_true)
    z_pred_mean_all.append(z_latent_ensemble.mean(axis=0))
    z_pred_std_all.append(np.abs(z_latent_ensemble).std(axis=0))
    y_pred_mean_all.append(y_pred_ensemble.mean(axis=0))
    y_pred_std_all.append(y_pred_ensemble.std(axis=0))
    
    del y_pred_ensemble, z_latent_ensemble, I_pred_ensemble

I_pred_mean_all = np.array(I_pred_mean_all)
I_pred_std_all = np.array(I_pred_std_all)
I_true_all = np.array(I_true_all)
z_pred_mean_all = np.array(z_pred_mean_all)
z_pred_std_all = np.array(z_pred_std_all)
y_pred_mean_all = np.array(y_pred_mean_all)
y_pred_std_all = np.array(y_pred_std_all)

print(f"Forecasting complete\n")

# Save forecast results
forecast_save_path = os.path.join(CHECKPOINTS_DIR, f'dmd_forecasts_rank{rank}_bootstrap{N_boot_strap}_thin{thin}_window{window}_years2-34.pkl')

forecast_results = {
    'I_pred_mean': I_pred_mean_all,
    'I_pred_std': I_pred_std_all,
    'I_true': I_true_all,
    'z_pred_mean': z_pred_mean_all,
    'z_pred_std': z_pred_std_all,
    'y_pred_mean': y_pred_mean_all,
    'y_pred_std': y_pred_std_all,
    'years': years,
    't_test': t_test,
    'metadata': dmd_filtered_results['metadata']
}

with open(forecast_save_path, 'wb') as f:
    dill.dump(forecast_results, f)

print(f"Forecasts saved: {forecast_save_path}\n")

# ============================================================================
# VISUALIZE LAST YEAR SPATIAL PREDICTIONS
# ============================================================================
print("Generating spatial visualization for last year...")
last_year_idx = -1
year = years[last_year_idx]

L_s_last = L_tot[last_year_idx]
Psi_s_last = Psi_tot[last_year_idx]
bn_s_last = bn_tot[last_year_idx]

y_pred_last_mean = y_pred_mean_all[last_year_idx]
y_pred_last_std = y_pred_std_all[last_year_idx]
y_true_last = data[year]

selected_days = [0, 91, 182, 273, 364]
day_labels = ['Day 0', 'Day 91', 'Day 182', 'Day 273', 'Day 364']

fig, axes = plt.subplots(3, len(selected_days), figsize=(24, 12))

for idx, (day, label) in enumerate(zip(selected_days, day_labels)):
    ax1 = axes[0, idx]
    im1 = ax1.contourf(x, y, y_true_last[day, :, :], levels=20, cmap='Blues')
    ax1.contour(x, y, mask_ice, levels=[0.5], colors='black', linewidths=0.5)
    ax1.set_title(f'{label}\nGround Truth', fontsize=10)
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = axes[1, idx]
    im2 = ax2.contourf(x, y, y_pred_last_mean[day, :, :], levels=20, cmap='Blues')
    ax2.contour(x, y, mask_ice, levels=[0.5], colors='black', linewidths=0.5)
    ax2.set_title(f'DMD Prediction', fontsize=10)
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = axes[2, idx]
    error = y_pred_last_mean[day, :, :] - y_true_last[day, :, :]
    vmax = np.max(np.abs(error))
    im3 = ax3.contourf(x, y, error, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax3.contour(x, y, mask_ice, levels=[0.5], colors='black', linewidths=0.5)
    ax3.set_title(f'Error', fontsize=10)
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

fig.suptitle(f'DMD Spatial Predictions for Year {year}', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'spatial_predictions_year{year}.png'), dpi=150, bbox_inches='tight')
plt.close()

# Uncertainty visualization
fig, axes = plt.subplots(1, len(selected_days), figsize=(24, 4))

for idx, (day, label) in enumerate(zip(selected_days, day_labels)):
    ax = axes[idx]
    im = ax.contourf(x, y, y_pred_last_std[day, :, :], levels=20, cmap='YlOrRd')
    ax.contour(x, y, mask_ice, levels=[0.5], colors='black', linewidths=0.5)
    ax.set_title(f'{label}\nUncertainty (Std)', fontsize=10)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(f'Bootstrap Uncertainty for Year {year}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'uncertainty_year{year}.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# TIME SERIES VISUALIZATION
# ============================================================================
print("Generating time series plots...")
I_pred_last = np.trapezoid(np.trapezoid(y_pred_last_mean, y, axis=2), x, axis=1)
I_true_last_year = np.trapezoid(np.trapezoid(y_true_last, y, axis=2), x, axis=1)

plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.plot(t_test, I_true_last_year, label='Ground Truth', color='black', linewidth=2.5)
plt.plot(t_test, I_pred_last, label='DMD Prediction', color='red', linestyle='--', linewidth=2)

I_pred_last_std_1d = np.trapezoid(np.trapezoid(y_pred_last_std, y, axis=2), x, axis=1)
plt.fill_between(t_test, I_pred_last - I_pred_last_std_1d, I_pred_last + I_pred_last_std_1d,
                 color='red', alpha=0.2, label='±1 Std')

plt.xlabel('Day of Year')
plt.ylabel('Total Ice Area')
plt.title(f'Integrated Ice Prediction for Year {year}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
error_integrated = I_pred_last - I_true_last_year
plt.plot(t_test, error_integrated, color='purple', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.fill_between(t_test, error_integrated, 0, where=(error_integrated >= 0), 
                 color='red', alpha=0.3, label='Over-prediction')
plt.fill_between(t_test, error_integrated, 0, where=(error_integrated < 0), 
                 color='blue', alpha=0.3, label='Under-prediction')

plt.xlabel('Day of Year')
plt.ylabel('Prediction Error')
plt.title(f'Prediction Error Over Year {year}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'integrated_ice_year{year}.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# LATENT COEFFICIENTS VISUALIZATION
# ============================================================================
print("Generating latent coefficient plots...")
fig, axes = plt.subplots(rank, 1, figsize=(16, 3.5 * rank))

for mode_idx in range(rank):
    ax = axes[mode_idx]
    
    for year_idx in range(len(years)):
        z_year = np.real(z_pred_mean_all[year_idx, :, mode_idx])
        ax.plot(t_test, z_year, linewidth=1, alpha=0.15, color='C0')
    
    z_mean_across_years = np.real(z_pred_mean_all[:, :, mode_idx]).mean(axis=0)
    z_std_across_years = np.real(z_pred_mean_all[:, :, mode_idx]).std(axis=0)
    
    ax.plot(t_test, z_mean_across_years, linewidth=3, color='darkblue', 
            label=f'Mean across {len(years)} years', zorder=10)
    ax.fill_between(t_test, z_mean_across_years - z_std_across_years,
                    z_mean_across_years + z_std_across_years,
                    alpha=0.3, color='darkblue', label='±1 Std', zorder=5)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Coefficient Amplitude')
    ax.set_title(f'DMD Latent Coefficient - Mode {mode_idx+1}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'latent_coefficients_all_modes.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# EIGENVALUE ANALYSIS
# ============================================================================
print("Generating eigenvalue analysis plots...")

# Histograms for each mode
for mode_idx in range(rank):
    plt.figure(figsize=(8, 4))
    plt.hist(L_s_last.imag[:, mode_idx], bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel('Imaginary part of eigenvalue')
    plt.ylabel('Count')
    plt.title(f'Distribution of Imaginary Eigenvalues - Mode {mode_idx+1}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, f'eigenvalue_imag_mode{mode_idx+1}.png'), dpi=150, bbox_inches='tight')
    plt.close()

# Scatter plots
plt.figure(figsize=(12, 4))
plt.plot(L_s_last.imag.flatten(), '.', alpha=0.5, markersize=4)
plt.xlabel('Index')
plt.ylabel('Imaginary part')
plt.title('Distribution of Imaginary Eigenvalues Across Bootstrap')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.savefig(os.path.join(RESULTS_DIR, 'eigenvalue_imag_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(L_s_last.real.flatten(), '.', alpha=0.5, markersize=4)
plt.xlabel('Index')
plt.ylabel('Real part')
plt.title('Distribution of Real Eigenvalues Across Bootstrap')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Stability boundary')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'eigenvalue_real_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# SPATIAL MODE VISUALIZATION
# ============================================================================
print("Generating spatial mode plots...")
L_avg = L_s_last.mean(axis=0)
bn_avg = bn_s_last.mean(axis=0)
Psi_avg = Psi_s_last.mean(axis=0)

fig, axes = plt.subplots(1, rank, figsize=(20, 4))

for mode_idx in range(rank):
    ax = axes[mode_idx]
    spatial_mode = np.abs(Psi_avg[mode_idx])
    
    im = ax.contourf(x, y, spatial_mode, levels=20, cmap='viridis')
    ax.contour(x, y, mask_ice, levels=[0.5], colors='white', linewidths=0.5)
    ax.set_title(f'Mode {mode_idx+1}', fontsize=11)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('DMD Spatial Mode Magnitudes', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'spatial_modes.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# SAVE SUMMARY STATISTICS
# ============================================================================
print("Saving summary statistics...")
median_eigs = np.median(L_s_last, axis=0)
median_imag = np.median(L_s_last.imag, axis=0)

summary_stats = {
    'n_years': len(years),
    'n_good_per_year': n_good_per_year.tolist(),
    'avg_good_per_year': float(n_good_per_year.mean()),
    'median_eigenvalues': median_eigs.tolist(),
    'median_imaginary_parts': median_imag.tolist(),
    'mode_statistics': []
}

for mode_idx in range(rank):
    z_mode_all = np.real(z_pred_mean_all[:, :, mode_idx]).flatten()
    summary_stats['mode_statistics'].append({
        'mode': mode_idx + 1,
        'mean': float(z_mode_all.mean()),
        'std': float(z_mode_all.std()),
        'min': float(z_mode_all.min()),
        'max': float(z_mode_all.max())
    })

with open(os.path.join(RESULTS_DIR, 'summary_statistics.pkl'), 'wb') as f:
    pickle.dump(summary_stats, f)

print("\n" + "="*80)
print("DMD ANALYSIS PIPELINE - COMPLETED")
print("="*80)
print(f"All results saved to: {RESULTS_DIR}")
print(f"All checkpoints saved to: {CHECKPOINTS_DIR}")