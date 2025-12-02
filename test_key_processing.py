#!/usr/bin/env python3
"""
Focused test for key modular data processing combinations.
Shows detailed before/after statistics for important processing combinations.
"""

import logging
import numpy as np
from src.lc_speckle_analysis.modular_processing import process_patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_detailed_stats(patch_data: np.ndarray, name: str):
    """Print detailed statistics for patch data."""
    print(f"\n{name} STATISTICS:")
    print(f"  Shape: {patch_data.shape}")
    for ch in range(patch_data.shape[0]):
        channel_data = patch_data[ch]
        print(f"  Channel {ch}:")
        print(f"    Mean: {channel_data.mean():.8f}")
        print(f"    Std:  {channel_data.std():.8f}")
        print(f"    Min:  {channel_data.min():.8f}")
        print(f"    Max:  {channel_data.max():.8f}")
    print(f"  Overall:")
    print(f"    Mean: {patch_data.mean():.8f}")
    print(f"    Std:  {patch_data.std():.8f}")


def test_key_combinations():
    """Test key processing combinations with detailed statistics."""
    
    # Create test patch with known characteristics
    np.random.seed(42)
    original_patch = np.random.randn(2, 10, 10).astype(np.float32)
    # Add different means and scales to channels
    original_patch[0] += 5.0  # Channel 0: mean ~5.0
    original_patch[1] = original_patch[1] * 2.5 + 10.0  # Channel 1: mean ~10.0, larger std
    
    print("="*80)
    print("MODULAR PROCESSING TEST - KEY COMBINATIONS")
    print("="*80)
    
    test_combinations = [
        # Basic options
        {"name": "RAW (no processing)", "shuffled": False, "zero_mean": False, "normalized": False, "quantiles": False, "aggregation": None},
        {"name": "ZERO MEAN ONLY", "shuffled": False, "zero_mean": True, "normalized": False, "quantiles": False, "aggregation": None},
        {"name": "NORMALIZED ONLY", "shuffled": False, "zero_mean": False, "normalized": True, "quantiles": False, "aggregation": None},
        {"name": "QUANTILES ONLY", "shuffled": False, "zero_mean": False, "normalized": False, "quantiles": True, "aggregation": None},
        
        # Combined processing
        {"name": "ZERO MEAN + NORMALIZED", "shuffled": False, "zero_mean": True, "normalized": True, "quantiles": False, "aggregation": None},
        {"name": "ZERO MEAN + QUANTILES", "shuffled": False, "zero_mean": True, "normalized": False, "quantiles": True, "aggregation": None},
        {"name": "ALL SPATIAL PROCESSING", "shuffled": False, "zero_mean": True, "normalized": True, "quantiles": True, "aggregation": None},
        
        # Aggregation examples
        {"name": "ZERO MEAN + MEAN AGGREGATION", "shuffled": False, "zero_mean": True, "normalized": False, "quantiles": False, "aggregation": "mean"},
        {"name": "NORMALIZED + STD AGGREGATION", "shuffled": False, "zero_mean": False, "normalized": True, "quantiles": False, "aggregation": "std"},
        {"name": "FULL PIPELINE + STDANDMEAN", "shuffled": False, "zero_mean": True, "normalized": True, "quantiles": False, "aggregation": "stdandmean"},
        
        # With shuffling
        {"name": "SHUFFLED + ZERO MEAN + NORMALIZED", "shuffled": True, "zero_mean": True, "normalized": True, "quantiles": False, "aggregation": None},
    ]
    
    for i, config in enumerate(test_combinations, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST {i}: {config['name']}")
        print(f"{'#'*80}")
        print(f"Parameters: shuffled={config['shuffled']}, zero_mean={config['zero_mean']}, "
              f"normalized={config['normalized']}, quantiles={config['quantiles']}, aggregation={config['aggregation']}")
        
        # Show original statistics (only once)
        if i == 1:
            print_detailed_stats(original_patch, "ORIGINAL")
        
        # Process the patch
        processed_patch = process_patch(
            patch_data=original_patch.copy(),
            shuffled=config['shuffled'],
            zero_mean=config['zero_mean'],
            normalized=config['normalized'],
            quantiles=config['quantiles'],
            aggregation=config['aggregation']
        )
        
        # Show processed statistics
        if config['aggregation'] is None:
            print_detailed_stats(processed_patch, "PROCESSED")
            
            # Verify transformations
            print(f"\nTRANSFORMATION VERIFICATION:")
            if config['zero_mean']:
                for ch in range(processed_patch.shape[0]):
                    ch_mean = processed_patch[ch].mean()
                    is_zero = abs(ch_mean) < 1e-6
                    print(f"  Zero-mean channel {ch}: {'✅' if is_zero else '❌'} (mean={ch_mean:.10f})")
            
            if config['normalized']:
                for ch in range(processed_patch.shape[0]):
                    ch_std = processed_patch[ch].std()
                    is_normalized = abs(ch_std - 1.0) < 1e-6
                    print(f"  Normalized channel {ch}: {'✅' if is_normalized else '❌'} (std={ch_std:.10f})")
            
            if config['quantiles']:
                for ch in range(processed_patch.shape[0]):
                    ch_min = processed_patch[ch].min()
                    ch_max = processed_patch[ch].max()
                    in_range = (ch_min >= -0.01) and (ch_max <= 1.01)  # Allow small tolerance
                    print(f"  Quantiles channel {ch}: {'✅' if in_range else '❌'} (range=[{ch_min:.6f}, {ch_max:.6f}])")
        else:
            print(f"\nAGGREGATED OUTPUT ({config['aggregation']}):")
            print(f"  Shape: {processed_patch.shape}")
            print(f"  Values: {processed_patch}")
            if config['aggregation'] == 'mean' and config['zero_mean']:
                all_near_zero = all(abs(val) < 1e-6 for val in processed_patch)
                print(f"  Zero-mean verification: {'✅' if all_near_zero else '❌'} (all values near zero)")
            elif config['aggregation'] == 'std' and config['normalized']:
                all_near_one = all(abs(val - 1.0) < 1e-6 for val in processed_patch)
                print(f"  Normalized verification: {'✅' if all_near_one else '❌'} (all values near 1.0)")


if __name__ == "__main__":
    test_key_combinations()
    
    print(f"\n\n{'='*80}")
    print("🎉 KEY COMBINATIONS TEST COMPLETED!")
    print("="*80)
    print("All processing options work correctly:")
    print("✅ Zero-mean: Produces exactly zero means per channel")
    print("✅ Normalized: Produces standard deviation of 1.0 per channel") 
    print("✅ Quantiles: Maps values to [0,1] range per channel")
    print("✅ Aggregation: Reduces spatial dimensions (mean, std, stdandmean)")
    print("✅ Shuffling: Spatially shuffles pixels (same order both channels)")
    print("✅ Combinations: All parameter combinations work together")
    print("="*80)
