# Project Structure 📂

> **Visual guide to the project organization**

---

## 🌳 Directory Tree

```
dietary-health-ml/
│
├── 📋 README.md                          # Main project overview
├── 📋 SESSION_SUMMARY.md                 # Complete session documentation (33KB)
├── 📋 QUICK_START_GUIDE.md              # Quick reference guide
├── 📋 PROJECT_SUMMARY.md                # Ver1 vs Ver2 explanation
├── 📋 PROJECT_STRUCTURE.md              # This file
│
├── 📁 data/                             # Data files (not in git)
│   ├── total_again.xlsx                 # Original data (29,098 visits)
│   └── ver2_paired_visits.csv           # Ver2 preprocessed data (to be generated)
│
├── 📁 result/                           # Results and outputs (not in git)
│   ├── ver1/                            # Ver1 model outputs
│   │   ├── models/                      # Trained models (.pth, .pkl)
│   │   ├── predictions/                 # Prediction results
│   │   └── visualizations/              # Performance plots
│   └── ver2_eda/                        # Ver2 EDA visualizations (to be generated)
│       ├── time_gap_distribution.png
│       ├── weight_change_distribution.png
│       └── correlation_heatmap.png
│
├── 📁 docs/                             # Documentation
│   ├── ANALYSIS_REPORT.md               # Ver1 comprehensive analysis (16KB)
│   └── INPUT_OUTPUT_EXPLANATION.md      # Feature explanations (17.7KB)
│
├── 📁 ver1/                             # Ver1: Cross-sectional Analysis ✅
│   ├── README.md                        # Ver1 methodology and limitations
│   ├── run_training.py                  # Interactive menu (4.4KB)
│   ├── train.bat                        # Windows batch file
│   ├── predict.bat                      # Prediction batch file
│   ├── visualize.bat                    # Visualization batch file
│   └── src/                             # Ver1 source code
│       ├── TABNET_ENHANCED_MODEL.py     # Main TabNet model (57.5KB)
│       ├── STACKING_ENSEMBLE_MODEL.py   # Ensemble model (24.8KB)
│       ├── EWMA_FEATURES.py             # Feature engineering (12.3KB)
│       ├── OPTUNA_STUDY.py              # Hyperparameter tuning (18.6KB)
│       ├── VISUALIZE_RESULTS.py         # Visualization (15.4KB)
│       ├── PREDICT_NEW_DATA.py          # Prediction (8.9KB)
│       └── MODEL_INTERPRETABILITY.py    # Interpretability (22.1KB)
│
└── 📁 ver2/                             # Ver2: Longitudinal Analysis 🚧
    ├── README.md                        # Ver2 development plan
    ├── data_preprocessing.py            # Paired visits creation (13.5KB)
    └── [models to be created]
        ├── xgboost_baseline.py          # Week 3-4
        ├── lstm_model.py                # Week 5-6
        └── transformer_model.py         # Week 5-6
```

---

## 🎯 File Classification

### 📘 Documentation Files

| File | Purpose | Audience | Size |
|------|---------|----------|------|
| `README.md` | Project overview | Everyone | 5KB |
| `SESSION_SUMMARY.md` | Complete session documentation | Development team | 33KB |
| `QUICK_START_GUIDE.md` | Quick reference | New users | 4KB |
| `PROJECT_SUMMARY.md` | Ver1/Ver2 explanation | Stakeholders | 6.7KB |
| `ver1/README.md` | Ver1 specifics | Ver1 users | 2.1KB |
| `ver2/README.md` | Ver2 development plan | Ver2 developers | 3.8KB |
| `docs/ANALYSIS_REPORT.md` | Ver1 analysis | Researchers | 16KB |
| `docs/INPUT_OUTPUT_EXPLANATION.md` | Feature explanations | Clinical users | 17.7KB |

**Total documentation**: ~88KB

---

### 💻 Source Code Files

#### Ver1 (Complete - 7 files)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `TABNET_ENHANCED_MODEL.py` | Main TabNet model | ~2,341 | ✅ Production |
| `STACKING_ENSEMBLE_MODEL.py` | Ensemble model | ~1,047 | ✅ Production |
| `EWMA_FEATURES.py` | Feature engineering | ~512 | ✅ Production |
| `OPTUNA_STUDY.py` | Hyperparameter tuning | ~783 | ✅ Production |
| `VISUALIZE_RESULTS.py` | Visualization | ~648 | ✅ Production |
| `PREDICT_NEW_DATA.py` | Prediction | ~374 | ✅ Production |
| `MODEL_INTERPRETABILITY.py` | Interpretability | ~928 | ✅ Production |

**Ver1 Total**: ~6,633 lines of Python code

#### Ver2 (In Development - 1 file)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `data_preprocessing.py` | Paired visits creation | ~450 | ✅ Ready to run |
| `xgboost_baseline.py` | Baseline model | - | ⏳ Week 3-4 |
| `lstm_model.py` | LSTM model | - | ⏳ Week 5-6 |
| `transformer_model.py` | Transformer model | - | ⏳ Week 5-6 |

**Ver2 Current**: ~450 lines (more to come)

---

### 🎛️ Execution Files

| File | Purpose | Platform | Version |
|------|---------|----------|---------|
| `ver1/run_training.py` | Interactive menu | Cross-platform | Ver1 |
| `ver1/train.bat` | Training script | Windows | Ver1 |
| `ver1/predict.bat` | Prediction script | Windows | Ver1 |
| `ver1/visualize.bat` | Visualization script | Windows | Ver1 |

---

### 📊 Data Files (Not in Git)

| File | Description | Size | Status |
|------|-------------|------|--------|
| `data/total_again.xlsx` | Original data | ~15MB | ✅ Available |
| `data/ver2_paired_visits.csv` | Ver2 preprocessed | ~3-5MB | ⏳ To be generated |

**Original Data Structure:**
- Rows: 29,098 visits
- Columns: ~50 (19 diet + 26 health + metadata)
- Format: Excel (.xlsx)

**Ver2 Data Structure:**
- Rows: ~18,000 paired visits
- Columns: ~142 (before + after + changes + derived)
- Format: CSV

---

### 📈 Result Files (Not in Git)

#### Ver1 Results (`result/ver1/`)

```
result/ver1/
├── models/
│   ├── tabnet_체중_best.pth
│   ├── tabnet_혈압_best.pth
│   └── ... (26 health indicators)
│
├── predictions/
│   ├── 체중_predictions.csv
│   └── ... (26 files)
│
└── visualizations/
    ├── 체중_learning_curve.png
    ├── 체중_prediction_scatter.png
    └── ... (52 files: 2 per indicator)
```

#### Ver2 Results (`result/ver2_eda/` - To be generated)

```
result/ver2_eda/
├── time_gap_distribution.png
├── weight_change_distribution.png
├── glucose_change_distribution.png
├── diet_health_scatter.png
└── correlation_heatmap.png
```

---

## 🔄 Data Flow Diagrams

### Ver1 Data Flow (Cross-sectional)

```
┌─────────────────┐
│ total_again.xlsx│
│  (29,098 rows)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load & Clean   │
│  - Handle NaN   │
│  - Normalize    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EWMA Features   │
│  - Trend        │
│  - Momentum     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train/Val/Test │
│   Split 70/15/15│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TabNet Model   │
│  or Stacking    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Predictions   │
│  + Metrics      │
│  + Visualize    │
└─────────────────┘
```

### Ver2 Data Flow (Longitudinal)

```
┌─────────────────┐
│ total_again.xlsx│
│  (29,098 rows)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  create_paired_visits() │
│  - Group by person_id   │
│  - Find consecutive     │
│  - Filter 30-365 days   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  calculate_changes()    │
│  - Δ diet features      │
│  - Δ health indicators  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  derived_features()     │
│  - Risk habit total     │
│  - Protective total     │
│  - Net improvement      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ ver2_paired_visits.csv  │
│   (~18,000 rows)        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Ver2 Models            │
│  - XGBoost (baseline)   │
│  - LSTM (advanced)      │
│  - Transformer (adv)    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Change Predictions     │
│  + Direction Accuracy   │
│  + R² for changes       │
└─────────────────────────┘
```

---

## 🎭 Model Architecture Comparison

### Ver1: TabNet Architecture

```
Input Layer (19 features: diet habits)
    │
    ▼
┌───────────────────────────────────┐
│  Sequential Attention Mechanism   │
│  - Feature Selection Step 1       │
│  - Feature Selection Step 2       │
│  - ... (N_steps)                  │
└───────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────┐
│  Feature Transformer Blocks       │
│  - Shared across steps            │
│  - GLU activations                │
│  - Skip connections               │
└───────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────┐
│  Output Layer                     │
│  - Single regression output       │
│  - Per health indicator           │
└───────────────────────────────────┘
    │
    ▼
Prediction: [체중 = 70.2kg]
```

**Key Parameters:**
- n_steps: 3-5
- n_d: 8-16 (decision dimension)
- n_a: 8-16 (attention dimension)
- gamma: 1.3 (relaxation parameter)

### Ver1: Stacking Ensemble Architecture

```
Input Layer (19 features)
    │
    ├─────┬─────┬─────┬─────┐
    │     │     │     │     │
    ▼     ▼     ▼     ▼     ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│XGBoost│ │LightGBM│ │CatBoost│ │Random│ │Gradient│
│      │ │      │ │      │ │Forest│ │Boosting│
└───┬──┘ └───┬──┘ └───┬──┘ └───┬──┘ └───┬──┘
    │        │        │        │        │
    └────────┴────────┴────────┴────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Ridge Regression │
        │   (Meta-learner)  │
        └────────┬──────────┘
                 │
                 ▼
         Final Prediction
```

### Ver2: Proposed LSTM Architecture (Week 5-6)

```
Input Layer (Before diet + After diet + Time gap)
    │
    ▼
┌───────────────────────────────────┐
│  LSTM Layer 1 (128 units)         │
│  - Captures temporal patterns     │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  Dropout (0.3)                    │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  LSTM Layer 2 (64 units)          │
│  - Higher-level abstractions      │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  Dense Layer (32 units)           │
│  - ReLU activation                │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  Output Layer (26 units)          │
│  - One per health indicator change│
└───────────┬───────────────────────┘
            │
            ▼
Prediction: [Δ체중 = -5.2kg, Δ혈당 = -12mg/dL, ...]
```

---

## 📊 Performance Tracking

### Ver1 Best Results (Achieved)

| Health Indicator | R² | RMSE | Clinical Status |
|------------------|----|----- |-----------------|
| 체중 (Weight) | 0.789 | 8.43kg | ⭐ Excellent |
| 수축기혈압 (SBP) | 0.214 | 14.23mmHg | ⚠️ Moderate |
| 이완기혈압 (DBP) | 0.176 | 9.87mmHg | ⚠️ Moderate |
| 공복혈당 (Glucose) | 0.123 | 18.45mg/dL | ⚠️ Weak |
| 총콜레스테롤 (Chol) | 0.099 | 28.67mg/dL | ⚠️ Weak |

**Interpretation:**
- ⭐ Excellent (R² > 0.7): Strong predictive power
- ✅ Good (R² 0.5-0.7): Useful predictions
- ⚠️ Moderate (R² 0.2-0.5): Limited utility
- ❌ Weak (R² < 0.2): Not clinically useful

### Ver2 Target Performance (Expected)

| Health Indicator Change | R² Target | Direction Accuracy | Development |
|------------------------|-----------|-------------------|-------------|
| Δ체중 | >0.65 | >75% | Week 5-6 |
| Δ혈당 | >0.55 | >70% | Week 5-6 |
| Δ콜레스테롤 | >0.45 | >65% | Week 5-6 |
| Δ혈압 | >0.40 | >65% | Week 5-6 |

**Direction Accuracy**: Percentage of times the model correctly predicts improvement vs decline

---

## 🔀 Version Comparison Summary

| Aspect | Ver1 (Cross-sectional) | Ver2 (Longitudinal) |
|--------|------------------------|---------------------|
| **Status** | ✅ Production-ready | 🚧 Development (Week 1) |
| **Data Rows** | 29,098 visits | ~18,000 pairs |
| **Input** | 19 diet features | 60+ features (before/after/change) |
| **Output** | 26 health values | 52 values (baseline + change) |
| **Prediction** | Diet → Health | Diet change → Health change |
| **R² (Weight)** | 0.789 (achieved) | >0.65 (target) |
| **Clinical Use** | Screening | Intervention planning |
| **Files** | 7 Python files | 1 file (+ 3 planned) |
| **Code Lines** | ~6,633 | ~450 (+ more to come) |
| **Documentation** | 3 files (35.8KB) | 1 file (3.8KB) |

---

## 🗺️ Navigation Guide

### For New Users:
1. Start with `QUICK_START_GUIDE.md`
2. Read `PROJECT_SUMMARY.md`
3. Choose Ver1 or Ver2 based on needs
4. Read version-specific README

### For Developers:
1. Read `SESSION_SUMMARY.md` (comprehensive)
2. Review `PROJECT_STRUCTURE.md` (this file)
3. Study source code in `ver1/src/` or `ver2/`
4. Check `docs/` for detailed analysis

### For Stakeholders:
1. Read `PROJECT_SUMMARY.md`
2. Review `docs/ANALYSIS_REPORT.md`
3. Check performance metrics above
4. Consult `ver1/README.md` and `ver2/README.md`

---

## 🎯 Development Checklist

### Ver1 (Complete)
- [x] TabNet model implementation
- [x] Stacking ensemble implementation
- [x] EWMA feature engineering
- [x] Optuna hyperparameter tuning
- [x] Visualization tools
- [x] Prediction tools
- [x] Interpretability tools
- [x] Interactive menu
- [x] Comprehensive documentation
- [x] Performance evaluation

### Ver2 (In Progress)
- [x] Data preprocessing script
- [x] Development plan
- [ ] Run preprocessing (Week 1)
- [ ] EDA analysis (Week 2)
- [ ] XGBoost baseline (Week 3-4)
- [ ] LSTM model (Week 5-6)
- [ ] Transformer model (Week 5-6)
- [ ] Performance evaluation (Week 7)
- [ ] Documentation (Week 8)
- [ ] Ver1 vs Ver2 comparison (Week 8)

---

## 📞 Quick Reference

**Run Ver1 Training:**
```bash
cd ver1
python run_training.py
```

**Run Ver2 Preprocessing:**
```bash
cd ver2
python data_preprocessing.py
```

**Generate Ver1 Predictions:**
```bash
cd ver1
python src/PREDICT_NEW_DATA.py
```

**View Ver1 Results:**
```bash
cd ver1
python src/VISUALIZE_RESULTS.py
```

---

**Last Updated**: 2025-11-05  
**Maintainer**: ML Development Team  
**Version**: 1.0
