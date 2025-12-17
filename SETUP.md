# SETUP GUIDE - Detailed Installation Instructions

## System Requirements

### Minimum
- OS: Windows, macOS, Linux
- Python: 3.8 or later
- RAM: 4 GB
- Disk Space: 2 GB
- Internet: Required (for API data download)

### Recommended
- Python: 3.10+
- RAM: 8+ GB
- GPU: NVIDIA (optional, for faster LSTM/GRU training)
- SSD: Faster data processing

## Step-by-Step Installation

### 1. Clone Repository

```bash
# Using Git
git clone https://github.com/yourusername/economic-crisis-prediction.git
cd economic-crisis-prediction

# Or download as ZIP
# Unzip and navigate to folder
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scriptsctivate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
# Basic installation
pip install -r requirements.txt

# With GPU support (NVIDIA CUDA)
pip install tensorflow[and-cuda]

# For Jupyter notebooks (optional)
pip install jupyter ipykernel
```

### 4. Register for API Keys

#### FRED (Federal Reserve Economic Data)

1. Visit https://fredaccount.stlouisfed.org/login
2. Click "Create New Account"
3. Enter email and create account
4. Verify email
5. Go to Settings > Account
6. Copy your API key (40 characters)
7. Save to environment or config file:

```bash
# Option A: Environment variable
export FRED_API_KEY=your_key_here  # macOS/Linux
set FRED_API_KEY=your_key_here     # Windows

# Option B: In config/fred_api.txt
echo your_key_here > config/fred_api.txt
```

#### World Bank (No key needed)

World Bank API is free and doesn't require authentication.

#### Kaggle (Optional, for extra data)

1. Visit https://www.kaggle.com/
2. Login or create account
3. Go to Settings > Account
4. Click "Create New Token"
5. This downloads `kaggle.json`
6. Place in `~/.kaggle/kaggle.json`:

```bash
# macOS/Linux
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle```

### 5. Create Necessary Directories

```bash
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p logs
mkdir -p reports/figures
mkdir -p reports/tables
```

### 6. Verify Installation

```bash
# Test imports
python -c "import pandas; import sklearn; import xgboost; import tensorflow; print('All imports successful!')"

# Test FRED API
python scripts/align_csv.py --test-fred-key YOUR_KEY

# Check directory structure
tree .  # or: dir /s (Windows)
```

## Configuration

### Model Parameters (config/model_params.yaml)

```yaml
# Random Forest
random_forest:
  n_estimators: 200          # Number of trees
  max_depth: 15              # Tree depth limit
  min_samples_split: 10      # Min samples to split
  class_weight: 'balanced'   # Handle class imbalance

# XGBoost
xgboost:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05        # Lower = slower, more stable
  subsample: 0.8             # Use 80% of data per iteration
  scale_pos_weight: 5.67     # Recession class weight

# Gradient Boosting
gradient_boosting:
  n_estimators: 200
  learning_rate: 0.05
  max_depth: 5

# LSTM
lstm:
  layers: [128, 64, 32]      # Layer sizes
  dropout: 0.2               # Regularization
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15

# GRU
gru:
  layers: [128, 64, 32]
  dropout: 0.2
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
```

### Data Configuration (config/data_config.yaml)

```yaml
date_range:
  start: '2000-01-01'
  end: '2024-12-31'

data_split:
  train: 0.70        # 70% for training
  validation: 0.20   # 20% for validation
  test: 0.10         # 10% for testing

feature_engineering:
  lookback_months: 12            # 12-month lookback
  moving_averages: [3, 6, 12]   # MA windows
  lag_features: [1, 2, 3, 6, 12] # Lag periods

missing_values:
  method: 'forward_fill'  # Or: 'backward_fill', 'interpolate'
  max_fill_gap: 6         # Max months to fill
```

### FRED Series (config/fred_series.json)

```json
{
  "gdp": "GDPC1",
  "unemployment": "UNRATE",
  "inflation": "CPIAUCSL",
  "oil_prices": "DCOILWTICO",
  "fed_funds_rate": "FEDFUNDS",
  "mortgage_rate": "MORTGAGE30US",
  "yield_curve": "T10Y2Y"
}
```

## Common Installation Issues

### Issue: "ModuleNotFoundError: No module named 'xgboost'"

**Solution:**
```bash
pip install xgboost
# or
pip install -r requirements.txt --force-reinstall
```

### Issue: "CUDA not available for GPU training"

**Solution:**
CUDA is optional. CPU training works fine but is slower.

For GPU support:
```bash
# For NVIDIA GPUs
pip install tensorflow[and-cuda]

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Issue: "FRED API key invalid"

**Solution:**
1. Verify key at https://fredaccount.stlouisfed.org/login
2. Key should be 40 characters
3. Try resetting: Account Settings > Create New Key
4. Clear cache: `rm -rf ~/.fredapi_cache/`

### Issue: "Memory error during model training"

**Solution:**
Reduce batch size or data size:
```yaml
# In config/model_params.yaml
batch_size: 16  # Instead of 32
```

Or use data subset for testing:
```bash
python scripts/recession_prediction_pipeline.py --sample-size 0.5  # Use 50% of data
```

## Verification Checklist

After setup, verify everything works:

```bash
# ✓ Python version
python --version  # Should be 3.8+

# ✓ Virtual environment active
echo $VIRTUAL_ENV  # Should show path (empty if not active)

# ✓ Dependencies installed
pip list | grep -E "pandas|sklearn|xgboost|tensorflow"

# ✓ API key configured
echo $FRED_API_KEY  # Should show key (if set as env var)

# ✓ Directory structure
ls -la data/  # Should show raw, processed, models

# ✓ Test script execution
python scripts/align_csv.py --help  # Should show help text

# ✓ Import test
python -c "from scripts.align_csv import CSVAligner; print('Success!')"
```

## Post-Installation

### Option 1: Quick Test

```bash
# Download sample data (small dataset)
python scripts/align_csv.py --fred-key YOUR_KEY --sample --output data/processed/

# Train models
python scripts/recession_prediction_pipeline.py --input data/processed/aligned_*.csv --output data/models/

# View results
cat reports/tables/model_comparison.csv
```

### Option 2: Full Run

```bash
# Download full dataset (25 years)
python scripts/align_csv.py --fred-key YOUR_KEY --auto-download --output data/processed/

# Train with cross-validation
python scripts/recession_prediction_pipeline.py \
  --input data/processed/aligned_*.csv \
  --output data/models/ \
  --cv-folds 5

# Generate report
cat reports/Economic_Crisis_Prediction.md
```

### Option 3: Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open and run notebooks/
# 01_data_exploration.ipynb → 04_results_analysis.ipynb
```

## System Specifications by Use Case

### Light Use (Testing Only)
- Python 3.8+
- 4 GB RAM
- 1 GB disk
- CPU only

### Standard Use (Training Models)
- Python 3.9+
- 8 GB RAM
- 2 GB disk
- CPU or GPU

### Production Deployment
- Python 3.10+
- 16+ GB RAM
- SSD (5+ GB)
- GPU strongly recommended

## Updating

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade xgboost

# Check for conflicts
pip check
```

## Uninstall

```bash
# Deactivate environment
deactivate

# Remove environment
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows

# Or if using conda
conda remove --name project-env --all
```

---

**Next Step:** See QUICK_START.md for 5-minute setup
