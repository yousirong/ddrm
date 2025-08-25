#!/bin/bash

echo "=== Setting up Optuna Optimization Environment ==="

# Install required Python packages
echo "Installing required packages..."
pip install optuna scikit-image opencv-python pillow

# Make the optimization script executable
chmod +x optuna_hyperparameter_optimization.py

echo ""
echo "=== Setup completed ==="
echo ""
echo "Usage:"
echo "1. Basic optimization (100 trials):"
echo "   python optuna_hyperparameter_optimization.py"
echo ""
echo "2. Custom number of trials:"
echo "   python optuna_hyperparameter_optimization.py --n_trials 50"
echo ""
echo "3. Custom paths:"
echo "   python optuna_hyperparameter_optimization.py \\"
echo "     --ddrm_path /path/to/ddrm \\"
echo "     --gt_path /path/to/ground_truth \\"
echo "     --n_trials 200"
echo ""
echo "Results will be saved in: ./optimization_results/"
echo "- optuna_study.db: Optuna study database"
echo "- optimization_results.json: Best parameters and score"
echo "- trial_XXXX.json: Individual trial results"
echo "- run_optimized_ddrm.sh: Script with best parameters"
echo ""
echo "To start optimization, run:"
echo "  python optuna_hyperparameter_optimization.py"