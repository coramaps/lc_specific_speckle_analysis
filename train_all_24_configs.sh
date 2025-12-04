#!/bin/bash

# Train all 24 configurations with nohup
# Each training will create its own directory with hash-based naming

PROJECT_DIR="/home/davideidmann/code/lc_specific_speckle_analysis"
cd "$PROJECT_DIR"

# List of all 24 configuration files
configs=(
    "config_base.conf"
    "config_base_mean.conf"
    "config_base_quantiles.conf"
    "config_base_shuffled.conf"
    "config_base_std.conf"
    "config_base_stdandmean.conf"
    "config_normalized.conf"
    "config_normalized_mean.conf"
    "config_normalized_quantiles.conf"
    "config_normalized_shuffled.conf"
    "config_normalized_std.conf"
    "config_normalized_stdandmean.conf"
    "config_zeromean.conf"
    "config_zeromean_mean.conf"
    "config_zeromean_normalized.conf"
    "config_zeromean_normalized_mean.conf"
    "config_zeromean_normalized_quantiles.conf"
    "config_zeromean_normalized_shuffled.conf"
    "config_zeromean_normalized_std.conf"
    "config_zeromean_normalized_stdandmean.conf"
    "config_zeromean_quantiles.conf"
    "config_zeromean_shuffled.conf"
    "config_zeromean_std.conf"
    "config_zeromean_stdandmean.conf"
)

echo "Starting training for all 24 configurations..."
echo "Training outputs will be saved to data/training_output/<hash>/"
echo "Log files will be saved as logs/train_<config_name>.log"

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to train a single config
train_config() {
    local config_file=$1
    local config_name=$(basename "$config_file" .conf)
    local log_file="logs/train_${config_name}.log"
    
    echo "Starting training for $config_name..."
    
    # Copy config to default location
    cp "configs/$config_file" "data/config.conf"
    
    # Run training with nohup
    nohup poetry run python -m src.lc_speckle_analysis.train_model > "$log_file" 2>&1 &
    
    # Get the PID of the background process
    local pid=$!
    echo "Training $config_name started with PID $pid (log: $log_file)"
    
    # Wait for this training to complete before starting the next one
    wait $pid
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Training $config_name completed successfully"
    else
        echo "❌ Training $config_name failed with exit code $exit_code"
    fi
    
    # Small delay between trainings
    sleep 2
}

# Train each configuration sequentially
for config in "${configs[@]}"; do
    train_config "$config"
done

echo ""
echo "All 24 configurations training completed!"
echo "Check data/training_output/ for results with proper hash-based naming"
echo "Check logs/ directory for individual training logs"

# Create summary of results
echo ""
echo "Training output directories created:"
ls -la data/training_output/
