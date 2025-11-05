# Quick Start Guide 🚀

> **Latest Status**: Ver1 complete, Ver2 preprocessing ready

---

## 📊 Choose Your Version

### Ver1: Cross-sectional Analysis (현재 사용 가능)
**Use when**: Screening, risk assessment, correlation analysis  
**Predicts**: Current diet → Current health  
**Status**: ✅ Production ready

```bash
cd ver1
python run_training.py
# Choose from interactive menu
```

---

### Ver2: Longitudinal Analysis (개발 중)
**Use when**: Intervention planning, change prediction  
**Predicts**: Diet changes → Health changes  
**Status**: 🚧 Data preprocessing ready

```bash
cd ver2
python data_preprocessing.py
# Generates paired visit data for Ver2 models
```

---

## 🎯 Quick Decision Tree

```
Do you want to...

├─ Know who is at high risk?
│  └─ Use Ver1 ✅
│
├─ Predict if someone will improve with diet changes?
│  └─ Use Ver2 (run preprocessing first) 🚧
│
├─ Screen large populations?
│  └─ Use Ver1 ✅
│
└─ Plan personalized interventions?
   └─ Use Ver2 (under development) 🚧
```

---

## 📁 Key Files Reference

| File | Purpose | Size | Status |
|------|---------|------|--------|
| `SESSION_SUMMARY.md` | Complete session documentation | 33KB | ✅ |
| `PROJECT_SUMMARY.md` | Ver1 vs Ver2 explanation | 6.7KB | ✅ |
| `ver1/README.md` | Ver1 methodology | 2.1KB | ✅ |
| `ver2/README.md` | Ver2 development plan | 3.8KB | ✅ |
| `ver2/data_preprocessing.py` | Ver2 data pipeline | 13.5KB | ✅ |
| `docs/ANALYSIS_REPORT.md` | Ver1 analysis report | 16KB | ✅ |
| `docs/INPUT_OUTPUT_EXPLANATION.md` | Feature explanations | 17.7KB | ✅ |

---

## ⚡ Next Immediate Action

**For Ver2 Development:**

```bash
# Step 1: Run preprocessing
cd ver2
python data_preprocessing.py

# Expected output:
# - ../data/ver2_paired_visits.csv (~18,000 rows)
# - ../result/ver2_eda/*.png (visualizations)
# - Summary statistics

# Step 2: Review EDA results
# Check ../result/ver2_eda/ for insights

# Step 3: Proceed to model development
# (LSTM, Transformer, or XGBoost models)
```

---

## 📈 Performance Comparison

| Metric | Ver1 (체중) | Ver2 Target (체중 변화) |
|--------|-------------|------------------------|
| R² | 0.789 | >0.65 |
| What it means | Strong correlation | Good change prediction |
| Clinical use | Risk screening | Intervention planning |

---

## ❓ FAQ

### Q1: Why two versions?
**A**: Ver1 shows correlation (who is healthy?), Ver2 predicts causation (who will improve?)

### Q2: Which version should I use?
**A**: 
- Screening/assessment → Ver1
- Intervention planning → Ver2 (when ready)

### Q3: Can Ver1 predict if changing habits will improve health?
**A**: ❌ No. Ver1 only shows current associations, not future changes.

### Q4: When will Ver2 be ready?
**A**: 8-week development plan:
- Week 1-2: Data preprocessing (current step)
- Week 3-4: Baseline XGBoost model
- Week 5-6: Advanced LSTM/Transformer models
- Week 7-8: Evaluation and documentation

### Q5: Do I need to keep Ver1?
**A**: ✅ Yes! Ver1 is valuable for:
- Quick risk screening
- Population-level analysis
- Baseline comparisons for Ver2

---

## 🔗 Related Documents

- **Full session details**: `SESSION_SUMMARY.md`
- **Why reorganization**: `PROJECT_SUMMARY.md`
- **Ver1 details**: `ver1/README.md`
- **Ver2 plan**: `ver2/README.md`
- **Ver1 analysis**: `docs/ANALYSIS_REPORT.md`

---

## 💡 Key Takeaways

1. ✅ **Ver1 is production-ready** for cross-sectional analysis
2. 🚧 **Ver2 preprocessing is ready** to run
3. 📊 **Clear distinction** between correlation (Ver1) and causation (Ver2)
4. 📚 **Comprehensive documentation** for all aspects
5. 🎯 **8-week roadmap** for Ver2 development

---

## 🆘 Need Help?

1. Check `SESSION_SUMMARY.md` (comprehensive)
2. Check version-specific READMEs
3. Review `PROJECT_SUMMARY.md` for big picture
4. Consult analysis reports in `docs/`

---

**Last Updated**: 2025-11-05  
**Project Status**: Ver1 complete ✅, Ver2 preprocessing ready 🚧
