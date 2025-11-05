# Changelog 📝

> **Project Evolution Timeline**

---

## [Ver2-Prep] - 2025-11-05

### 🎯 Major Milestone: Project Reorganization

**Critical Discovery:**
- User identified fundamental mismatch between expectations and Ver1 implementation
- Ver1 performs Cross-sectional analysis (correlation), not Longitudinal analysis (causation)
- Decision made to preserve Ver1 and develop Ver2 for true change prediction

### ✨ Added

#### Ver2 Foundation
- `ver2/` folder structure
- `ver2/README.md` - Comprehensive Ver2 development plan (8-week roadmap)
- `ver2/data_preprocessing.py` (13.5KB) - Complete paired visits creation pipeline
  - `create_paired_visits()` function
  - `calculate_derived_features()` function
  - `perform_eda()` function
  - Full preprocessing pipeline

#### Ver1 Preservation
- `ver1/` folder structure
- `ver1/README.md` - Clear explanation of Ver1 methodology and limitations
- Moved all existing Ver1 code to `ver1/src/`
- Moved execution files to `ver1/` (run_training.py, *.bat)

#### Comprehensive Documentation
- `SESSION_SUMMARY.md` (33KB) - Complete session documentation
  - 4-phase session evolution
  - Ver1 vs Ver2 detailed comparison
  - All problems solved
  - Ver2 development roadmap
  - Key learning points
- `QUICK_START_GUIDE.md` (4KB) - Quick reference for users
  - Decision tree (Ver1 vs Ver2)
  - Quick commands
  - FAQ section
- `PROJECT_SUMMARY.md` (6.7KB) - Why reorganization was needed
- `PROJECT_STRUCTURE.md` (14KB) - Complete project visualization
  - Directory tree with file sizes
  - Data flow diagrams
  - Model architecture diagrams
  - Performance tracking
- `CHANGELOG.md` (this file) - Project evolution timeline

### 🔄 Changed

#### Main README
- Added version comparison section at top
- Updated project structure with Ver1/Ver2 folders
- Added Ver1 vs Ver2 comparison table

### 📊 Impact

**Before This Update:**
- Single version (confusing cross-sectional vs longitudinal)
- Ambiguous documentation about causation
- No clear path for change prediction development

**After This Update:**
- Clear separation: Ver1 (correlation) vs Ver2 (causation)
- Ver1 preserved and production-ready
- Ver2 development path established with 8-week plan
- Comprehensive documentation (88KB total)
- Clear clinical use cases for each version

---

## [Documentation] - 2025-11-05 (Earlier)

### ✨ Added

#### Analysis Reports
- `docs/ANALYSIS_REPORT.md` (16KB) - Ver1 comprehensive analysis
  - 10-section detailed report
  - Executive summary
  - Methodology explanation
  - Results and performance metrics
  - Clinical implications
  - Limitations and future directions

- `docs/INPUT_OUTPUT_EXPLANATION.md` (17.7KB) - Feature explanations
  - 19 dietary habit features explained
  - 26 health indicators explained
  - 3 realistic case studies with predictions
  - Clinical interpretation guide
  - Korean and English terminology

### 📊 Impact
- Total documentation: 33.7KB
- Provides complete understanding of Ver1 model
- Clinical context for predictions
- Real-world usage examples

---

## [Python Interface] - 2025-11-05 (Earlier)

### ✨ Added

#### Interactive Training Menu
- `run_training.py` (4.4KB) - Python-based training interface
  - Menu options:
    1. TabNet training (single target)
    2. Stacking Ensemble training (single target)
    3. Process all targets (automated batch)
    4. Exit
  - Automatic working directory detection
  - Korean language interface
  - GPU/CPU auto-selection
  - Progress tracking

### 🐛 Fixed

#### File Path Issues
- **Problem**: `run_training.py` couldn't find `../data/total_again.xlsx`
- **Solution**: Automatic working directory setup
  ```python
  script_dir = os.path.dirname(os.path.abspath(__file__))
  os.chdir(script_dir)
  ```
- **Result**: Works consistently from any execution location

### 📊 Impact
- User can run training without batch files
- More intuitive for Python users
- Better error handling and user feedback

---

## [Critical Fixes] - 2025-11-05 (Earlier)

### 🐛 Fixed

#### 1. TabNetWrapper sklearn Compatibility
- **Problem**: TabNetWrapper not recognized as valid sklearn regressor
  ```python
  ValueError: TabNetWrapper is not a valid sklearn regressor
  ```
- **Root Cause**: Missing sklearn base classes and required methods
- **Solution**: Proper inheritance and method implementation
  ```python
  from sklearn.base import BaseEstimator, RegressorMixin
  
  class TabNetWrapper(BaseEstimator, RegressorMixin):
      def get_params(self, deep=True):
          return {"tabnet_model": self.tabnet_model}
      
      def set_params(self, **params):
          if "tabnet_model" in params:
              self.tabnet_model = params["tabnet_model"]
              self.model = params["tabnet_model"]
          return self
  ```
- **Result**: ✅ TabNetWrapper now works with StackingRegressor

#### 2. Git Desktop.ini Corruption
- **Problem**: 
  ```bash
  fatal: bad object refs/desktop.ini
  fatal: unable to read tree 8f0e3c2a1b...
  ```
- **Root Cause**: Windows created Desktop.ini in .git/refs/
- **Solution**:
  1. Added to .gitignore
  2. Removed from git tracking
  3. Hard reset to origin/main
  4. Clean working directory
- **Result**: ✅ Git repository restored and functional

### 📊 Impact
- Models can now use sklearn stacking ensemble
- Git workflow restored
- Better compatibility with sklearn ecosystem

---

## [Initial Setup] - 2025-11-05 (Earlier)

### ✅ Confirmed

#### Development Environment
- **OS**: Windows 10/11
- **Python**: 3.8+
- **GPU**: CUDA 12.4/13.0 compatible
- **Git Workflow**: Pull only (no push permissions)

#### Existing Code Review
- TabNet model with attention mechanism
- Stacking ensemble (5 base models + Ridge meta-learner)
- EWMA feature engineering
- Optuna hyperparameter optimization
- Visualization tools
- Prediction tools
- Model interpretability tools

### 🧹 Cleaned

#### File Cleanup
- Removed excessive duplicate files
- Organized project structure
- Removed temporary files
- Updated .gitignore

### 📊 Impact
- Clean project structure
- Clear understanding of codebase
- Ready for development work

---

## Version Comparison Timeline

| Version | Date | Purpose | Status | Key Features |
|---------|------|---------|--------|--------------|
| **Ver1** | Before 2025-11-05 | Cross-sectional analysis | ✅ Complete | TabNet, Stacking, R²=0.789 |
| **Ver2-Prep** | 2025-11-05 | Longitudinal development | 🚧 Week 1 | Data preprocessing ready |

---

## Performance Evolution

### Ver1 Best Results (Achieved)

| Target | Initial | After Tuning | Improvement |
|--------|---------|-------------|-------------|
| 체중 (Weight) | R²=0.65 | R²=0.789 | +21.4% |
| 수축기혈압 (SBP) | R²=0.15 | R²=0.214 | +42.7% |
| 공복혈당 (Glucose) | R²=0.08 | R²=0.123 | +53.8% |

### Ver2 Expected Progress

| Week | Milestone | Expected R² (Weight Change) |
|------|-----------|----------------------------|
| 1-2 | Data preprocessing | - |
| 3-4 | XGBoost baseline | >0.50 |
| 5-6 | LSTM/Transformer | >0.65 |
| 7-8 | Ensemble & tuning | >0.70 |

---

## Documentation Evolution

| Date | Document | Size | Purpose |
|------|----------|------|---------|
| 2025-11-05 | SESSION_SUMMARY.md | 33KB | Complete session documentation |
| 2025-11-05 | QUICK_START_GUIDE.md | 4KB | Quick reference |
| 2025-11-05 | PROJECT_SUMMARY.md | 6.7KB | Ver1 vs Ver2 explanation |
| 2025-11-05 | PROJECT_STRUCTURE.md | 14KB | Project visualization |
| 2025-11-05 | CHANGELOG.md | 6KB | This file |
| 2025-11-05 (Earlier) | ANALYSIS_REPORT.md | 16KB | Ver1 analysis |
| 2025-11-05 (Earlier) | INPUT_OUTPUT_EXPLANATION.md | 17.7KB | Feature explanations |
| Earlier | README.md | 5KB | Main overview |

**Total Documentation**: ~102KB

---

## Code Statistics Evolution

### Ver1 Development

| Phase | Files | Lines | Features |
|-------|-------|-------|----------|
| Initial | 3 files | ~3,000 | Basic TabNet |
| Mid | 5 files | ~5,000 | + Stacking, EWMA |
| Final | 7 files | ~6,633 | + Optuna, interpretability |

### Ver2 Development (Planned)

| Phase | Files | Lines | Features |
|-------|-------|-------|----------|
| Week 1-2 | 1 file | ~450 | Data preprocessing |
| Week 3-4 | 2 files | ~1,500 | + XGBoost baseline |
| Week 5-6 | 4 files | ~3,500 | + LSTM, Transformer |
| Week 7-8 | 5 files | ~5,000 | + Ensemble, documentation |

---

## Key Learnings Timeline

### 2025-11-05 (Critical Discovery)
**Lesson**: Cross-sectional ≠ Longitudinal
- User question revealed fundamental misunderstanding
- Ver1 does correlation, not causation
- Solution: Separate Ver1 and Ver2 with clear documentation

### 2025-11-05 (Earlier - Technical)
**Lesson**: sklearn compatibility requirements
- Custom estimators need BaseEstimator, RegressorMixin
- Must implement get_params/set_params
- Enables sklearn ecosystem integration

### 2025-11-05 (Earlier - Workflow)
**Lesson**: Windows git issues
- Desktop.ini can corrupt git refs
- Always include in .gitignore
- Hard reset may be needed for recovery

---

## Future Roadmap

### Week 1-2: Data Preprocessing & EDA
- [x] Create data_preprocessing.py
- [ ] Run preprocessing on Windows
- [ ] Analyze EDA results
- [ ] Document data quality

### Week 3-4: Baseline Model
- [ ] Implement XGBoost baseline
- [ ] Hyperparameter tuning
- [ ] Evaluate performance
- [ ] Create baseline report

### Week 5-6: Advanced Models
- [ ] Implement LSTM
- [ ] Implement Transformer
- [ ] Compare models
- [ ] Ensemble best models

### Week 7-8: Evaluation & Documentation
- [ ] Cross-validation
- [ ] Clinical interpretation
- [ ] Ver2 analysis report
- [ ] Ver1 vs Ver2 comparison
- [ ] Deployment guide

### Post-Ver2 (Future)
- [ ] Multi-timepoint analysis (3+ visits)
- [ ] Personalized recommendations
- [ ] Subgroup analysis
- [ ] Causal inference methods
- [ ] Web API deployment
- [ ] Mobile application

---

## Breaking Changes

### 2025-11-05 Major Reorganization

**What Changed:**
- All Ver1 code moved to `ver1/` folder
- New `ver2/` folder created
- File paths updated in execution scripts

**Migration Guide:**

**Before:**
```bash
# Old structure
project/
├── src/
│   └── TABNET_ENHANCED_MODEL.py
└── run_training.py
```

**After:**
```bash
# New structure
project/
├── ver1/
│   ├── src/
│   │   └── TABNET_ENHANCED_MODEL.py
│   └── run_training.py
└── ver2/
    └── data_preprocessing.py
```

**User Action Required:**
- Update any custom scripts to use `ver1/` paths
- Review ver1/README.md for Ver1-specific usage
- Check ver2/README.md for Ver2 development plan

---

## Acknowledgments

### Contributors
- ML Development Team
- Clinical Consultation Team
- Data Science Team

### Technologies Used
- **Deep Learning**: PyTorch, PyTorch-TabNet
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Optimization**: Optuna
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Version Control**: Git, GitHub

---

## Related Documents

- `SESSION_SUMMARY.md` - Complete session documentation
- `QUICK_START_GUIDE.md` - Quick reference
- `PROJECT_SUMMARY.md` - Ver1 vs Ver2 explanation
- `PROJECT_STRUCTURE.md` - Project visualization
- `README.md` - Main overview

---

**Last Updated**: 2025-11-05  
**Current Version**: Ver1 (Complete), Ver2 (Week 1 of 8)  
**Next Milestone**: Ver2 data preprocessing execution
