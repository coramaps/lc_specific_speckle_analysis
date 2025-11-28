#!/usr/bin/env python3
"""
Utility script to retroactively save config JSON for existing training runs.
This ensures all runs have their configuration saved for reconstructability.
"""

import sys
import json
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.lc_speckle_analysis.data_config import TrainingDataConfig

def save_config_for_existing_run(run_directory: Path, config: TrainingDataConfig, config_hash: str):
    """Save config JSON for an existing training run."""
    
    # Convert config to dictionary
    config_dict = asdict(config)
    
    # Add metadata
    config_dict['_metadata'] = {
        'config_hash': config_hash,
        'generated_at': datetime.now().isoformat(),
        'retroactively_added': True,
        'note': 'Configuration retroactively added for existing training run'
    }
    
    # Save to JSON file in the run directory
    config_file = run_directory / f"config_{config_hash}.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"✅ Configuration saved to: {config_file}")

def main():
    """Save config for the existing c78a6a9e run."""
    # Load the original config (without data_with_zero_mean to match the original hash)
    config_path = project_root / "data" / "config.conf"
    
    # Temporarily modify config loading to get the original configuration
    import configparser
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    
    # Create config without data_with_zero_mean to match original hash c78a6a9e
    original_config = TrainingDataConfig.from_file(config_path)
    original_config.data_with_zero_mean = False  # This was the original setting
    
    # Verify the hash matches
    original_hash = original_config.get_config_hash()
    print(f"Original config hash: {original_hash}")
    
    if original_hash == "c78a6a9e":
        # Save config for existing run
        run_dir = project_root / "data" / "training_output" / f"run_{original_hash}"
        if run_dir.exists():
            save_config_for_existing_run(run_dir, original_config, original_hash)
        else:
            print(f"❌ Run directory not found: {run_dir}")
    else:
        print(f"❌ Hash mismatch. Expected c78a6a9e, got {original_hash}")

if __name__ == "__main__":
    main()
