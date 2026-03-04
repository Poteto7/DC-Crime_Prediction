# 🚨 D.C. Crime Radar: Spatial-Temporal Crime Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end machine learning pipeline predicting weekly crime hotspots and forecasting incident volumes across 46 Washington D.C. neighborhood clusters.**

---

## 🧭 1. Project Overview

This project tackles urban safety through two distinct predictive modeling approaches using real-world data from the D.C. Metropolitan Police Department (2021–2024):

1. **Classification:** Will a given neighborhood cluster become a crime "hotspot" next week?
2. **Forecasting:** How many crimes will occur in that cluster next week?

---

## 🗂️ 2. Repository Structure

```
dc-crime-radar/
│
├── notebooks/
│   ├── 01_data_pipeline_and_classification.ipynb   # Cleaning, feature engineering, RF & XGBoost
│   └── 02_lstm_forecasting.ipynb                   # LSTM time-series forecasting
│
├── report/
│   └── crime_prediction_report.pdf
│
├── requirements.txt
└── README.md
```

---

## 🏗️ 3. Architecture

### Full Pipeline

```
  DC MPD Raw Crime Data (89,719 records · 2021–2023 train | 29,289 records · 2024 holdout)
                          │
                          ▼
          ┌───────────────────────────────────┐
          │          DATA PIPELINE            │
          │  · Drop irrelevant cols (CCN,     │
          │    BLOCK, CENSUS_TRACT, etc.)     │
          │  · Drop nulls in critical cols    │
          │    (START_DATE, LAT, LON,         │
          │    DISTRICT, CLUSTER)             │
          │  · Filter to date range           │
          │  · Extract: year, week_of_year    │
          │  · One-hot encode: OFFENSE,       │
          │    SHIFT, METHOD, DISTRICT        │
          │    (vocabulary saved to disk —    │
          │    reused on 2024 without refit)  │
          │                                   │
          │  Train output: 88,933 clean rows  │
          └──────────────┬────────────────────┘
                         │
                         ▼
          ┌───────────────────────────────────┐
          │        FEATURE ENGINEERING        │
          │  · Weekly aggregation by cluster  │
          │    (46 clusters saved to disk)    │
          │  · Hotspot label: crime_count > 20│
          │  · Lag features: lag1, lag4       │
          │  · Rolling mean & std (4-week)    │
          │                                   │
          │  Train output: 7,314 weekly rows  │
          └──────────────┬────────────────────┘
                         │
            Strict Chronological Split
            Train: 2021–2023 | Holdout: 2024
            (2024 not loaded during training)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
  ┌─────────────────────┐  ┌──────────────────────┐
  │   TASK 1            │  │   TASK 2              │
  │   Hotspot           │  │   Crime Count         │
  │   Classification    │  │   Forecasting         │
  │                     │  │                       │
  │  · Random Forest    │  │  · LSTM (Keras)       │
  │  · XGBoost          │  │  · 4-week sliding     │
  │                     │  │    window per cluster │
  │  Handles 21.2%      │  │  · MinMaxScaler       │
  │  hotspot base rate  │  │    fitted on train,   │
  │  (3.67:1 imbalance) │  │    saved to disk      │
  └─────────────────────┘  └──────────────────────┘
```

### LSTM Model Architecture

```
  Input per cluster: 4-week sliding window
  ┌──────────────────────────────────────────────────┐
  │  X = [ week_t-4, week_t-3, week_t-2, week_t-1 ] │
  │        scaled crime_count (MinMaxScaler)          │
  │        scale range: [0.0, 85.0] crimes           │
  └───────────────────┬──────────────────────────────┘
                      │  shape: (batch, 4, 1)
                      ▼
          ┌───────────────────────┐
          │      LSTM Layer       │
          │      units  =  50     │
          │   activation = ReLU   │
          │    params = 10,400    │
          └───────────┬───────────┘
                      │  shape: (batch, 50)
                      ▼
          ┌───────────────────────┐
          │    Dropout Layer      │
          │      rate   = 0.2     │
          │    params   =   0     │
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │      Dense Layer      │
          │      units  =  1      │
          │  activation = linear  │
          │      params =  51     │
          └───────────┬───────────┘
                      │
                      ▼
          ŷ (scaled) → inverse_transform()
                      │
                      ▼
          Predicted crime count (original scale)

  Total trainable params: 10,451
  Training: 50 epochs · batch_size=32 · val_split=0.1
  Optimizer: Adam · Loss: MSE
  Callback: EarlyStopping(patience=10, restore_best_weights=True)
```

---

## 📊 4. Models & Key Results

**Strict holdout design: 2024 data not loaded anywhere during training. Models and scalers serialised to disk and evaluated on fully unseen 2024 data.**

### 🎯 Task 1 — Hotspot Classification

Predicts if a cluster will exceed **20 crimes** the following week. Both models tuned for a **3.67:1 class imbalance** (21.2% hotspot base rate).

| Model | Strategy | Accuracy | ROC-AUC |
|:---|:---|:---:|:---:|
| Random Forest | `class_weight='balanced_subsample'`, 100 estimators | **91%** | **0.9687** |
| XGBoost | `scale_pos_weight=3.67` | **91%** | **0.9653** |

**XGBoost — 2024 Holdout Classification Report:**

| Class | Precision | Recall | F1 |
|:---|:---:|:---:|:---:|
| Not Hotspot | 0.96 | 0.94 | 0.95 |
| Hotspot | 0.74 | 0.81 | 0.77 |
| **Weighted Avg** | **0.92** | **0.91** | **0.91** |

### 📈 Task 2 — Weekly Crime Volume Forecasting (LSTM)

Captures temporal trends using a 4-week sliding window per cluster. Evaluated on 2024 holdout using the scaler fitted on 2021–2023 only.

| Metric | Score |
|:---|:---:|
| R² Score | **0.7914** |
| MAE | **~3.4 crimes** |

---

## 🔧 5. Feature Engineering

| Feature | Description |
|:---|:---|
| `year`, `week_of_year` | ISO calendar week extracted from `START_DATE` |
| `crime_count_lag1` | Crime count from the previous week per cluster |
| `crime_count_lag4` | Crime count from 4 weeks prior per cluster |
| `crime_count_roll_mean4` | 4-week rolling average per cluster |
| `crime_count_roll_std4` | 4-week rolling standard deviation per cluster |
| `is_hotspot_next_week` | Binary classification target: `crime_count_next_week > 20` |

**Leakage prevention:** OHE vocabulary, cluster set, MinMaxScaler, and model weights are all fitted on 2021–2023 and serialised to disk. The 2024 pipeline loads these artefacts — no refitting on test data.

---

## 📁 6. Dataset

**Source:** [DC Open Data — Crime Incidents](https://opendata.dc.gov/datasets/crime-incidents-in-2024)

| Detail | Value |
|:---|:---|
| Training raw records (2021–2023) | 89,719 |
| Training after cleaning | 88,933 |
| Holdout raw records (2024) | 29,289 |
| Neighborhood clusters | 46 |
| Weekly training rows (aggregated) | 7,314 |

> Raw files are not included due to size. Download from DC Open Data and run the notebooks in order to regenerate all artefacts.

---

## ⚙️ 7. Getting Started

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/SaiSrinivasGuttikonda/dc-crime-radar.git
cd dc-crime-radar

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebooks in order
#    Notebook 1 trains RF & XGBoost and serialises models to disk
jupyter notebook notebooks/01_data_pipeline_and_classification.ipynb

#    Notebook 2 trains LSTM on 2021-2023, then evaluates on 2024 holdout
jupyter notebook notebooks/02_lstm_forecasting.ipynb
```

---

## 👤 8. Author

**Sai Srinivas Guttikonda** — M.S. Computer Science, George Washington University (2026)  
[LinkedIn](#) · [Portfolio](#)
