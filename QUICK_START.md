# QUICK START GUIDE - Economic Crisis Prediction

## 5-Minute Setup

### 1. Prerequisites
```bash
# Check Python version (3.8+)
python --version

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scriptsctivate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get API Keys

**FRED (Federal Reserve):**
1. Visit https://fredaccount.stlouisfed.org/login
2. Register for free account
3. Copy your API key

**World Bank:**
- No key required (free API)

**Kaggle:**
1. Visit https://www.kaggle.com/settings/account
2. Click "Create New Token"
3. Save `kaggle.json` to `~/.kaggle/`

### 4. Download Data (Choose One)

**Option A: Automatic (Easiest)**
```bash
python scripts/align_csv.py --fred-key YOUR_FRED_KEY --auto-download
```

**Option B: Manual**
1. Download CSVs from:
   - FRED: https://fred.stlouisfed.org/ (search series IDs: GDPC1, UNRATE, etc.)
   - World Bank: https://data.worldbank.org/ (search indicators)
   - Kaggle: https://www.kaggle.com/ (search "economic crisis" or "recession")
2. Save to `data/raw/`
3. Run: `python scripts/align_csv.py --csv-dir ./data/raw/`

### 5. Train Models
```bash
python scripts/recession_prediction_pipeline.py \
  --input data/processed/aligned_economic_indicators.csv \
  --output data/models/
```

### 6. View Results
```bash
# Model performance
cat reports/tables/model_comparison.csv

# Feature importance
cat reports/tables/feature_importance_ranking.csv

# Full paper (convert to PDF if needed)
cat reports/Economic_Crisis_Prediction.md
```

---

## Command Reference

### CSV Alignment Script

```bash
# Basic usage
python scripts/align_csv.py

# With FRED API key (auto-download)
python scripts/align_csv.py --fred-key YOUR_KEY --auto-download

# With custom CSV directory
python scripts/align_csv.py --csv-dir ./my_data/

# With output directory
python scripts/align_csv.py --output ./results/

# With missing value handling method
python scripts/align_csv.py --missing-method forward_fill  # Options: forward_fill, interpolate, drop

# Full options
python scripts/align_csv.py --help
```

### ML Training Pipeline

```bash
# Basic usage
python scripts/recession_prediction_pipeline.py

# Specify input/output
python scripts/recession_prediction_pipeline.py \
  --input data/processed/aligned_*.csv \
  --output data/models/

# Cross-validation folds
python scripts/recession_prediction_pipeline.py --cv-folds 10

# Save detailed metrics
python scripts/recession_prediction_pipeline.py --save-metrics detailed_metrics.csv

# Enable GPU (for LSTM/GRU)
python scripts/recession_prediction_pipeline.py --gpu

# Full options
python scripts/recession_prediction_pipeline.py --help
```

---

## Expected Output

After running `align_csv.py`:
```
✓ Loaded 6 CSV files
✓ Aligned to common date range: 2000-01-01 to 2024-12-31
✓ Handled missing values: Forward-fill method
✓ Output: aligned_economic_indicators.csv
✓ Report: data_validation_report.txt
```

After running `recession_prediction_pipeline.py`:
```
================================================================================
ALGORITHM 1: RANDOM FOREST
================================================================================
Accuracy: 0.8345
Precision: 0.7932
Recall: 0.8012
F1-Score: 0.7971
ROC-AUC: 0.8742

... (4 more algorithms) ...

================================================================================
MODEL COMPARISON SUMMARY
================================================================================
                 Accuracy  Precision    Recall  F1-Score   ROC-AUC  Matthews CC
Random Forest       0.8345     0.7932    0.8012    0.7971    0.8742       0.6523
XGBoost             0.8621     0.8432    0.8189    0.8308    0.8987       0.7234
Gradient Boosting   0.8543     0.8123    0.7921    0.8020    0.8834       0.6987
LSTM                0.8412     0.8001    0.8134    0.8066    0.8654       0.6789
GRU                 0.8498     0.8211    0.8001    0.8104    0.8723       0.6923
Ensemble (Voting)   0.8734     0.8632    0.8389    0.8509    0.9087       0.7412
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'xgboost'` | Run `pip install xgboost` |
| `fredapi error: Invalid API key` | Check key at https://fredaccount.stlouisfed.org/login |
| `FileNotFoundError: aligned_economic_indicators.csv` | Run `align_csv.py` first |
| `MemoryError during LSTM training` | Reduce batch size in `config/model_params.yaml` |
| `No GPU found` | This is OK; CPU training works (slower) |

See `docs/TROUBLESHOOTING.md` for more.

---

## Next Steps

1. **Understand the methodology:**
   - Read `docs/METHODOLOGY.md` for technical details
   - Read `reports/Training_Metrics_Report.txt` for best practices

2. **Review results:**
   - Open `reports/Economic_Crisis_Prediction.md` (full research paper)
   - Check `reports/figures/` for visualizations

3. **Customize for your use:**
   - Edit `config/model_params.yaml` to tune hyperparameters
   - Edit `config/data_config.yaml` for different date ranges/splits
   - Add new indicators in `config/fred_series.json`

4. **Deploy/extend:**
   - Deploy as web app: `notebooks/04_results_analysis.ipynb`
   - Add real-time predictions: See `scripts/utils/`
   - Integrate into larger system: All scripts are modular

---

## Files You'll Use Most

| File | Purpose |
|------|---------|
| `scripts/align_csv.py` | Download & align economic data |
| `scripts/recession_prediction_pipeline.py` | Train all 5 ML algorithms |
| `config/model_params.yaml` | Adjust hyperparameters |
| `reports/Economic_Crisis_Prediction.md` | Full research write-up |
| `reports/tables/model_comparison.csv` | View model metrics |

---

**Ready to start?** Run:
```bash
python scripts/align_csv.py --fred-key YOUR_KEY --auto-download
python scripts/recession_prediction_pipeline.py
```

**Questions?** See README.md or docs/TROUBLESHOOTING.md
