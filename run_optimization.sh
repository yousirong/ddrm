#!/bin/bash

echo "=== Ultrasound DDRM Hyperparameter Optimization ==="
echo ""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --version VERSION     Target version to optimize (V3, V4, V5, V6, V7, ALL)"
    echo "                        Default: V3"
    echo "  --trials N            Number of trials per version (default: 30)"
    echo "  --basic               Run basic optimization (all versions together)"
    echo "  --setup               Setup environment and install requirements"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --setup                    # Setup environment first"
    echo "  $0 --version V3 --trials 50   # Optimize V3 with 50 trials"
    echo "  $0 --version ALL --trials 20  # Optimize all versions with 20 trials each"
    echo "  $0 --basic --trials 100       # Basic optimization with 100 trials"
    echo ""
}

# Default values
VERSION="V3"
TRIALS=30
BASIC=false
SETUP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --basic)
            BASIC=true
            shift
            ;;
        --setup)
            SETUP=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Setup environment
if [ "$SETUP" = true ]; then
    echo "Setting up optimization environment..."
    bash setup_optimization.sh
    exit 0
fi

# Validate version
if [ "$VERSION" != "ALL" ] && [ "$VERSION" != "V3" ] && [ "$VERSION" != "V4" ] && [ "$VERSION" != "V5" ] && [ "$VERSION" != "V6" ] && [ "$VERSION" != "V7" ]; then
    echo "Error: Invalid version '$VERSION'. Must be V3, V4, V5, V6, V7, or ALL."
    exit 1
fi

# Check if required files exist
if [ ! -f "optuna_hyperparameter_optimization.py" ]; then
    echo "Error: optuna_hyperparameter_optimization.py not found"
    echo "Make sure you're running this script from the DDRM directory"
    exit 1
fi

if [ ! -f "optuna_version_specific_optimization.py" ]; then
    echo "Error: optuna_version_specific_optimization.py not found"
    exit 1
fi

# Check if Python packages are installed
python -c "import optuna, cv2, skimage" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required Python packages not installed."
    echo "Run: $0 --setup"
    exit 1
fi

# Create output directory for logs
mkdir -p optimization_logs

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ "$BASIC" = true ]; then
    echo "Starting basic hyperparameter optimization..."
    echo "Trials: $TRIALS"
    echo "Log file: optimization_logs/basic_optimization_${TIMESTAMP}.log"
    echo ""
    
    python optuna_hyperparameter_optimization.py \
        --n_trials $TRIALS \
        2>&1 | tee optimization_logs/basic_optimization_${TIMESTAMP}.log
    
else
    echo "Starting version-specific hyperparameter optimization..."
    echo "Version: $VERSION"
    echo "Trials: $TRIALS"
    echo "Log file: optimization_logs/${VERSION}_optimization_${TIMESTAMP}.log"
    echo ""
    
    python optuna_version_specific_optimization.py \
        --version $VERSION \
        --n_trials $TRIALS \
        2>&1 | tee optimization_logs/${VERSION}_optimization_${TIMESTAMP}.log
fi

# Check if optimization completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Optimization Completed Successfully ==="
    echo ""
    
    if [ "$BASIC" = true ]; then
        echo "Results saved in: optimization_results/"
        if [ -f "optimization_results/run_optimized_ddrm.sh" ]; then
            echo "Optimized script: optimization_results/run_optimized_ddrm.sh"
            echo ""
            echo "To run with optimized parameters:"
            echo "  cd optimization_results && ./run_optimized_ddrm.sh"
        fi
    else
        if [ "$VERSION" = "ALL" ]; then
            echo "Results saved in: optimization_results_V*/"
            echo "Combined results: optimization_results_all_versions.json"
            echo ""
            echo "Optimized scripts available:"
            for v in V3 V4 V5 V6 V7; do
                if [ -f "optimization_results_${v}/run_optimized_ddrm_${v}.sh" ]; then
                    echo "  ${v}: optimization_results_${v}/run_optimized_ddrm_${v}.sh"
                fi
            done
        else
            echo "Results saved in: optimization_results_${VERSION}/"
            if [ -f "optimization_results_${VERSION}/run_optimized_ddrm_${VERSION}.sh" ]; then
                echo "Optimized script: optimization_results_${VERSION}/run_optimized_ddrm_${VERSION}.sh"
                echo ""
                echo "To run with optimized parameters:"
                echo "  cd optimization_results_${VERSION} && ./run_optimized_ddrm_${VERSION}.sh"
            fi
        fi
    fi
    
    echo ""
    echo "You can also view the optimization progress using Optuna dashboard:"
    echo "  optuna-dashboard sqlite:///optimization_results*/optuna_study*.db"
    
else
    echo ""
    echo "=== Optimization Failed ==="
    echo "Check the log file for details: optimization_logs/*_${TIMESTAMP}.log"
    exit 1
fi