"""
Trend Prediction for Antarctic Climate Variables
=================================================

This script implements two approaches for predicting trends in three Antarctic variables:
1. SARIMA (Seasonal AutoRegressive Integrated Moving Average)
2. Simple LSTM (Long Short-Term Memory Neural Network)

Variables analyzed:
- Ice Thickness (m)
- Sea Surface Temperature (°C)
- Ice Concentration (fraction)

Author: Generated for Antarctic climate analysis
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Statistical and time series libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Deep learning libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Data loading
import dill
import sys
sys.path.append('./src/modules')
from data_wrangle import thin_data, del_leap

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("TREND PREDICTION: SARIMA vs LSTM")
print("=" * 80)

# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[1/6] Loading data...")

# Load the three datasets
with open('./data/ice/Antarctic_SST_1993_2023.pkl', 'rb') as f:
    sst_data_, sst_years_ = dill.load(f)

with open('./data/ice/Antarctic_thickness_1993_2023.pkl', 'rb') as f:
    thickness_data_, thickness_years_ = dill.load(f)

with open('./data/ice/Antarctic_years_1989_2024i.pkl', 'rb') as f:
    mask_land_, mask_ice_, data_, data_mean_month_, data_mean_week_, x_, y_ = dill.load(f)

# Preprocessing
thin = 1
data, data_mean_month, data_mean_week, x, y, mask_ice, mask_land = \
    thin_data(thin, data_, data_mean_month_, data_mean_week_, x_, y_, mask_ice_, mask_land_)

data = del_leap(data)
thickness_data = thin_data(thin, thickness_data_)[0]
sst_data = thin_data(thin, sst_data_)[0]
thickness_data = del_leap(thickness_data)
sst_data = del_leap(sst_data)

# Use data from 1993-2023 (31 years)
data = data[4:-1]
print(f"✓ Loaded {len(data)} years of data (1993-2023)")

# Compute spatially averaged timeseries
def spatial_mean_timeseries(data, mask):
    """Compute spatial mean ignoring masked pixels"""
    n_time = data.shape[0]
    ts = np.zeros(n_time)
    for t in range(n_time):
        frame = data[t, :, :]
        valid = ~mask & ~np.isnan(frame)
        ts[t] = np.nanmean(frame[valid])
    return ts

def stack_years(data_list):
    """Stack list of yearly arrays into single continuous array"""
    return np.concatenate(data_list, axis=0)

# Stack all years and compute spatial means
thickness_full = stack_years(thickness_data)
sst_full = stack_years(sst_data)
ice_full = stack_years(data)

thickness_ts = spatial_mean_timeseries(thickness_full, mask_land)
sst_ts = spatial_mean_timeseries(sst_full, mask_land)
ice_ts = spatial_mean_timeseries(ice_full, mask_land)

print(f"✓ Created time series: {len(thickness_ts)} daily observations")

# Create time index
dates = pd.date_range(start='1993-01-01', periods=len(thickness_ts), freq='D')

# Create DataFrames for each variable
df_thickness = pd.DataFrame({'date': dates, 'value': thickness_ts})
df_sst = pd.DataFrame({'date': dates, 'value': sst_ts})
df_ice = pd.DataFrame({'date': dates, 'value': ice_ts})

df_thickness.set_index('date', inplace=True)
df_sst.set_index('date', inplace=True)
df_ice.set_index('date', inplace=True)

print(f"✓ Data preparation complete")
print(f"  - Ice Thickness range: [{thickness_ts.min():.3f}, {thickness_ts.max():.3f}] m")
print(f"  - SST range: [{sst_ts.min():.3f}, {sst_ts.max():.3f}] °C")
print(f"  - Ice Concentration range: [{ice_ts.min():.3f}, {ice_ts.max():.3f}]")

# ============================================================================
# SECTION 2: TRAIN-TEST SPLIT
# ============================================================================

print("\n[2/6] Splitting data into train/test sets...")

# Use 80% for training, 20% for testing
train_size = int(0.8 * len(thickness_ts))
test_size = len(thickness_ts) - train_size

train_dates = dates[:train_size]
test_dates = dates[train_size:]

# Split data
thickness_train, thickness_test = thickness_ts[:train_size], thickness_ts[train_size:]
sst_train, sst_test = sst_ts[:train_size], sst_ts[train_size:]
ice_train, ice_test = ice_ts[:train_size], ice_ts[train_size:]

print(f"✓ Train set: {train_size} days ({train_size/365:.1f} years)")
print(f"✓ Test set: {test_size} days ({test_size/365:.1f} years)")

# ============================================================================
# SECTION 3: SARIMA MODEL
# ============================================================================

print("\n[3/6] Training SARIMA models...")
print("  (This may take several minutes per variable)")

def fit_sarima_model(train_data, order, seasonal_order, name):
    """
    Fit SARIMA model to training data
    
    Args:
        train_data: Training time series
        order: (p, d, q) ARIMA order
        seasonal_order: (P, D, Q, s) seasonal order
        name: Variable name for display
    
    Returns:
        Fitted SARIMA model
    """
    print(f"\n  Training SARIMA for {name}...")
    print(f"    Order: {order}, Seasonal: {seasonal_order}")
    
    try:
        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = model.fit(disp=False, maxiter=200)
        print(f"    ✓ AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
        return results
    
    except Exception as e:
        print(f"    ✗ Error fitting SARIMA: {e}")
        return None

# SARIMA parameters
# For daily data with yearly seasonality: s=365
# Start with simple parameters and adjust if needed
p, d, q = 1, 1, 1  # ARIMA order
P, D, Q, s = 1, 1, 1, 365  # Seasonal order

sarima_order = (p, d, q)
sarima_seasonal = (P, D, Q, s)

# Fit SARIMA models for each variable
sarima_thickness = fit_sarima_model(
    thickness_train, sarima_order, sarima_seasonal, "Ice Thickness"
)

sarima_sst = fit_sarima_model(
    sst_train, sarima_order, sarima_seasonal, "SST"
)

sarima_ice = fit_sarima_model(
    ice_train, sarima_order, sarima_seasonal, "Ice Concentration"
)

# Make predictions
print("\n  Making SARIMA predictions...")

def sarima_predict(model, train_data, n_forecast):
    """Generate predictions from SARIMA model"""
    if model is None:
        return np.full(n_forecast, np.nan)
    
    try:
        forecast = model.forecast(steps=n_forecast)
        return forecast.values
    except Exception as e:
        print(f"    Error in prediction: {e}")
        return np.full(n_forecast, np.nan)

sarima_thickness_pred = sarima_predict(sarima_thickness, thickness_train, test_size)
sarima_sst_pred = sarima_predict(sarima_sst, sst_train, test_size)
sarima_ice_pred = sarima_predict(sarima_ice, ice_train, test_size)

print("  ✓ SARIMA predictions complete")

# ============================================================================
# SECTION 4: LSTM MODEL
# ============================================================================

print("\n[4/6] Training LSTM models...")

# LSTM hyperparameters
SEQUENCE_LENGTH = 365  # Use 1 year of data to predict next point
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {device}")

class TimeSeriesDataset(Dataset):
    """Dataset for time series data with sliding window"""
    
    def __init__(self, data, sequence_length):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return x.unsqueeze(1), y  # Add feature dimension

class LSTMModel(nn.Module):
    """Simple LSTM model for time series prediction"""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from last time step
        out = self.fc(out[:, -1, :])
        return out

def train_lstm(train_data, sequence_length, epochs, name):
    """
    Train LSTM model
    
    Args:
        train_data: Training time series (numpy array)
        sequence_length: Length of input sequences
        epochs: Number of training epochs
        name: Variable name for display
    
    Returns:
        Trained model and scaler
    """
    print(f"\n  Training LSTM for {name}...")
    
    # Normalize data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    
    # Create dataset and dataloader
    dataset = TimeSeriesDataset(train_scaled, sequence_length)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = LSTMModel(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            
            # Forward pass
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    print(f"    ✓ Training complete")
    return model, scaler

def lstm_predict(model, scaler, train_data, n_forecast, sequence_length):
    """
    Generate predictions from LSTM model
    
    Args:
        model: Trained LSTM model
        scaler: StandardScaler used for normalization
        train_data: Training data (for initial sequence)
        n_forecast: Number of steps to forecast
        sequence_length: Length of input sequences
    
    Returns:
        Predictions array
    """
    model.eval()
    
    # Normalize training data
    train_scaled = scaler.transform(train_data.reshape(-1, 1)).flatten()
    
    # Start with last sequence from training data
    current_sequence = train_scaled[-sequence_length:].copy()
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_forecast):
            # Prepare input
            x = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(2).to(device)
            
            # Predict next value
            pred = model(x)
            pred_value = pred.cpu().numpy()[0, 0]
            predictions.append(pred_value)
            
            # Update sequence (rolling window)
            current_sequence = np.append(current_sequence[1:], pred_value)
    
    # Inverse transform to original scale
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions).flatten()
    
    return predictions

# Train LSTM models
lstm_thickness, scaler_thickness = train_lstm(
    thickness_train, SEQUENCE_LENGTH, EPOCHS, "Ice Thickness"
)

lstm_sst, scaler_sst = train_lstm(
    sst_train, SEQUENCE_LENGTH, EPOCHS, "SST"
)

lstm_ice, scaler_ice = train_lstm(
    ice_train, SEQUENCE_LENGTH, EPOCHS, "Ice Concentration"
)

# Make predictions
print("\n  Making LSTM predictions...")
lstm_thickness_pred = lstm_predict(
    lstm_thickness, scaler_thickness, thickness_train, test_size, SEQUENCE_LENGTH
)
lstm_sst_pred = lstm_predict(
    lstm_sst, scaler_sst, sst_train, test_size, SEQUENCE_LENGTH
)
lstm_ice_pred = lstm_predict(
    lstm_ice, scaler_ice, ice_train, test_size, SEQUENCE_LENGTH
)
print("  ✓ LSTM predictions complete")

# ============================================================================
# SECTION 5: EVALUATION
# ============================================================================

print("\n[5/6] Evaluating models...")

def evaluate_predictions(y_true, y_pred_sarima, y_pred_lstm, name):
    """
    Compute evaluation metrics
    
    Args:
        y_true: True values
        y_pred_sarima: SARIMA predictions
        y_pred_lstm: LSTM predictions
        name: Variable name
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n  {name}:")
    metrics = {}
    
    # SARIMA metrics
    if not np.any(np.isnan(y_pred_sarima)):
        sarima_rmse = np.sqrt(mean_squared_error(y_true, y_pred_sarima))
        sarima_mae = mean_absolute_error(y_true, y_pred_sarima)
        sarima_r2 = r2_score(y_true, y_pred_sarima)
        
        print(f"    SARIMA - RMSE: {sarima_rmse:.4f}, MAE: {sarima_mae:.4f}, R²: {sarima_r2:.4f}")
        
        metrics['sarima'] = {
            'rmse': sarima_rmse,
            'mae': sarima_mae,
            'r2': sarima_r2
        }
    else:
        print("    SARIMA - Failed to generate predictions")
        metrics['sarima'] = None
    
    # LSTM metrics
    lstm_rmse = np.sqrt(mean_squared_error(y_true, y_pred_lstm))
    lstm_mae = mean_absolute_error(y_true, y_pred_lstm)
    lstm_r2 = r2_score(y_true, y_pred_lstm)
    
    print(f"    LSTM   - RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}, R²: {lstm_r2:.4f}")
    
    metrics['lstm'] = {
        'rmse': lstm_rmse,
        'mae': lstm_mae,
        'r2': lstm_r2
    }
    
    return metrics

# Evaluate all variables
metrics_thickness = evaluate_predictions(
    thickness_test, sarima_thickness_pred, lstm_thickness_pred, "Ice Thickness"
)

metrics_sst = evaluate_predictions(
    sst_test, sarima_sst_pred, lstm_sst_pred, "SST"
)

metrics_ice = evaluate_predictions(
    ice_test, sarima_ice_pred, lstm_ice_pred, "Ice Concentration"
)

# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

print("\n[6/6] Creating visualizations...")

# Create comprehensive comparison plot
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

variables = [
    ('Ice Thickness', thickness_train, thickness_test, 
     sarima_thickness_pred, lstm_thickness_pred, 'm', 'blue'),
    ('SST', sst_train, sst_test, 
     sarima_sst_pred, lstm_sst_pred, '°C', 'red'),
    ('Ice Concentration', ice_train, ice_test, 
     sarima_ice_pred, lstm_ice_pred, '', 'cyan')
]

for idx, (name, train, test, sarima_pred, lstm_pred, unit, color) in enumerate(variables):
    ax = axes[idx]
    
    # Plot training data
    ax.plot(train_dates, train, color=color, alpha=0.5, linewidth=0.8, 
            label='Training data')
    
    # Plot test data (ground truth)
    ax.plot(test_dates, test, color=color, linewidth=1.5, 
            label='Test data (ground truth)')
    
    # Plot SARIMA predictions
    if not np.any(np.isnan(sarima_pred)):
        ax.plot(test_dates, sarima_pred, color='orange', linestyle='--', 
                linewidth=2, label='SARIMA prediction')
    
    # Plot LSTM predictions
    ax.plot(test_dates, lstm_pred, color='green', linestyle='--', 
            linewidth=2, label='LSTM prediction')
    
    # Formatting
    ax.set_ylabel(f'{name} ({unit})' if unit else name, fontsize=12)
    ax.set_title(f'{name} - SARIMA vs LSTM Predictions', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add vertical line at train/test split
    ax.axvline(x=test_dates[0], color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(test_dates[0], ax.get_ylim()[1], ' Test →', 
            verticalalignment='top', fontsize=9, alpha=0.7)

axes[-1].set_xlabel('Date', fontsize=12)
plt.tight_layout()
plt.savefig('trend_prediction_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved plot: trend_prediction_comparison.png")

# Create detailed zoom on test period
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, (name, train, test, sarima_pred, lstm_pred, unit, color) in enumerate(variables):
    ax = axes[idx]
    
    # Plot only test period
    ax.plot(test_dates, test, color=color, linewidth=2, 
            label='Actual', marker='o', markersize=2)
    
    if not np.any(np.isnan(sarima_pred)):
        ax.plot(test_dates, sarima_pred, color='orange', linestyle='--', 
                linewidth=2, label='SARIMA', marker='s', markersize=2)
    
    ax.plot(test_dates, lstm_pred, color='green', linestyle='--', 
            linewidth=2, label='LSTM', marker='^', markersize=2)
    
    ax.set_ylabel(f'{name} ({unit})' if unit else name, fontsize=12)
    ax.set_title(f'{name} - Test Period Detail', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Date', fontsize=12)
plt.tight_layout()
plt.savefig('trend_prediction_test_detail.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved plot: trend_prediction_test_detail.png")

# Create metrics comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_list = [metrics_thickness, metrics_sst, metrics_ice]
var_names = ['Ice Thickness', 'SST', 'Ice Concentration']

for idx, (metrics, name) in enumerate(zip(metrics_list, var_names)):
    ax = axes[idx]
    
    if metrics['sarima'] is not None:
        sarima_rmse = metrics['sarima']['rmse']
        sarima_r2 = metrics['sarima']['r2']
    else:
        sarima_rmse = 0
        sarima_r2 = 0
    
    lstm_rmse = metrics['lstm']['rmse']
    lstm_r2 = metrics['lstm']['r2']
    
    x = np.arange(2)
    width = 0.35
    
    # RMSE comparison
    ax_twin = ax.twinx()
    bars1 = ax.bar(x[0] - width/2, sarima_rmse, width, label='SARIMA', color='orange')
    bars2 = ax.bar(x[0] + width/2, lstm_rmse, width, label='LSTM', color='green')
    
    # R² comparison
    bars3 = ax_twin.bar(x[1] - width/2, sarima_r2, width, color='orange', alpha=0.7)
    bars4 = ax_twin.bar(x[1] + width/2, lstm_r2, width, color='green', alpha=0.7)
    
    ax.set_ylabel('RMSE', fontsize=11)
    ax_twin.set_ylabel('R²', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['RMSE', 'R²'])
    
    if idx == 0:
        ax.legend(loc='upper left', fontsize=9)
    
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved plot: metrics_comparison.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\n📊 Summary of Results:\n")

for name, metrics in [('Ice Thickness', metrics_thickness), 
                       ('SST', metrics_sst), 
                       ('Ice Concentration', metrics_ice)]:
    print(f"{name}:")
    
    if metrics['sarima'] is not None:
        print(f"  SARIMA: RMSE={metrics['sarima']['rmse']:.4f}, R²={metrics['sarima']['r2']:.4f}")
    else:
        print(f"  SARIMA: Failed")
    
    print(f"  LSTM:   RMSE={metrics['lstm']['rmse']:.4f}, R²={metrics['lstm']['r2']:.4f}")
    
    # Determine winner
    if metrics['sarima'] is not None:
        if metrics['lstm']['rmse'] < metrics['sarima']['rmse']:
            print(f"  🏆 Winner: LSTM (lower RMSE)")
        else:
            print(f"  🏆 Winner: SARIMA (lower RMSE)")
    else:
        print(f"  🏆 Winner: LSTM (by default)")
    print()

print("📁 Generated files:")
print("  - trend_prediction_comparison.png (full comparison)")
print("  - trend_prediction_test_detail.png (test period detail)")
print("  - metrics_comparison.png (performance metrics)")

print("\n✅ Script execution complete!")
print("=" * 80)
