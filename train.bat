@echo off
echo Starting training...
cd src
python -c "from TABNET_ENHANCED_MODEL import main; main(use_tabnet_stacking=True, use_optuna=False)"
pause
