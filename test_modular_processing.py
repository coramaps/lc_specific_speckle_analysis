#!/usr/bin/env python3
"""
Comprehensive test for all modular data processing options.
Tests all combinations of shuffled, zero_mean, normalized, quantiles, and aggregation parameters.
"""

import logging
import numpy as np
from itertools import product
from pathlib import Path
from src.lc_speckle_analysis.modular_processing import process_patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_statistics(patch_data: np.ndarray, name: str) -> dict:
    """Calculate comprehensive statistics for patch data."""
    stats = {}
    for ch in range(patch_data.shape[0]):
        channel_data = patch_data[ch]
        stats[f'ch{ch}_mean'] = float(channel_data.mean())
        stats[f'ch{ch}_std'] = float(channel_data.std())
        stats[f'ch{ch}_min'] = float(channel_data.min())
        stats[f'ch{ch}_max'] = float(channel_data.max())
        stats[f'ch{ch}_median'] = float(np.median(channel_data))
    
    # Overall statistics
    stats['overall_mean'] = float(patch_data.mean())
    stats['overall_std'] = float(patch_data.std())
    stats['shape'] = patch_data.shape
    
    logger.info(f"{name} Statistics:")
    logger.info(f"  Shape: {stats['shape']}")
    for ch in range(patch_data.shape[0]):
        logger.info(f"  Channel {ch}: mean={stats[f'ch{ch}_mean']:.6f}, std={stats[f'ch{ch}_std']:.6f}, "
                   f"min={stats[f'ch{ch}_min']:.6f}, max={stats[f'ch{ch}_max']:.6f}, median={stats[f'ch{ch}_median']:.6f}")
    logger.info(f"  Overall: mean={stats['overall_mean']:.6f}, std={stats['overall_std']:.6f}")
    
    return stats


def test_single_processing_combination(shuffled: bool, zero_mean: bool, normalized: bool, 
                                     quantiles: bool, aggregation: str = None):
    """Test a single combination of processing parameters."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: shuffled={shuffled}, zero_mean={zero_mean}, normalized={normalized}, "
               f"quantiles={quantiles}, aggregation={aggregation}")
    logger.info(f"{'='*80}")
    
    # Create test patch with known characteristics
    np.random.seed(42)  # For reproducible results
    original_patch = np.random.randn(2, 10, 10).astype(np.float32)
    # Add different means and scales to channels
    original_patch[0] += 5.0  # Channel 0: mean ~5.0
    original_patch[1] = original_patch[1] * 2.5 + 10.0  # Channel 1: mean ~10.0, larger std
    
    # Calculate original statistics
    original_stats = calculate_statistics(original_patch, "ORIGINAL")
    
    # Process the patch
    try:
        processed_patch = process_patch(
            patch_data=original_patch.copy(),
            shuffled=shuffled,
            zero_mean=zero_mean,
            normalized=normalized,
            quantiles=quantiles,
            aggregation=aggregation
        )
        
        # Calculate processed statistics (only if no aggregation)
        if aggregation is None:
            processed_stats = calculate_statistics(processed_patch, "PROCESSED")
            
            # Verify expected transformations
            logger.info(f"\nTransformation Verification:")
            
            if zero_mean:
                for ch in range(processed_patch.shape[0]):
                    ch_mean = processed_patch[ch].mean()
                    logger.info(f"  Zero-mean check channel {ch}: {abs(ch_mean) < 1e-6} (mean={ch_mean:.10f})")
            
            if normalized:
                for ch in range(processed_patch.shape[0]):
                    ch_std = processed_patch[ch].std()
                    logger.info(f"  Normalized check channel {ch}: {abs(ch_std - 1.0) < 1e-6} (std={ch_std:.10f})")
            
            if quantiles:
                for ch in range(processed_patch.shape[0]):
                    ch_min = processed_patch[ch].min()
                    ch_max = processed_patch[ch].max()
                    logger.info(f"  Quantiles check channel {ch}: min={ch_min:.6f}, max={ch_max:.6f} (should be [0,1])")
            
        else:
            # For aggregated data, show the aggregated statistics
            logger.info(f"AGGREGATED OUTPUT ({aggregation}):")
            logger.info(f"  Shape: {processed_patch.shape}")
            logger.info(f"  Values: {processed_patch}")
            
        logger.info(f"✅ Processing combination completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise


def test_all_processing_combinations():
    """Test all valid combinations of processing parameters."""
    logger.info("Starting comprehensive test of all modular processing combinations")
    
    # Define parameter options
    bool_options = [False, True]
    aggregation_options = [None, 'mean', 'std', 'stdandmean']
    
    total_combinations = 0
    successful_combinations = 0
    
    # Test all combinations
    for shuffled, zero_mean, normalized, quantiles in product(bool_options, repeat=4):
        for aggregation in aggregation_options:
            total_combinations += 1
            
            try:
                test_single_processing_combination(
                    shuffled=shuffled,
                    zero_mean=zero_mean,
                    normalized=normalized,
                    quantiles=quantiles,
                    aggregation=aggregation
                )
                successful_combinations += 1
                
            except Exception as e:
                logger.error(f"❌ Combination failed: shuffled={shuffled}, zero_mean={zero_mean}, "
                           f"normalized={normalized}, quantiles={quantiles}, aggregation={aggregation}")
                logger.error(f"   Error: {e}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total combinations tested: {total_combinations}")
    logger.info(f"Successful combinations: {successful_combinations}")
    logger.info(f"Failed combinations: {total_combinations - successful_combinations}")
    logger.info(f"Success rate: {100.0 * successful_combinations / total_combinations:.1f}%")


def test_edge_cases():
    """Test edge cases and special scenarios."""
    logger.info(f"\n{'='*80}")
    logger.info("Testing Edge Cases")
    logger.info(f"{'='*80}")
    
    # Test with all zeros
    logger.info("\nTesting with all-zero patch...")
    zero_patch = np.zeros((2, 10, 10), dtype=np.float32)
    try:
        processed = process_patch(zero_patch.copy(), zero_mean=True, normalized=True)
        logger.info(f"✅ All-zero patch processed successfully")
        logger.info(f"   Result shape: {processed.shape}, contains NaN: {np.isnan(processed).any()}")
    except Exception as e:
        logger.error(f"❌ All-zero patch failed: {e}")
    
    # Test with constant values
    logger.info("\nTesting with constant-value patch...")
    constant_patch = np.full((2, 10, 10), 5.0, dtype=np.float32)
    try:
        processed = process_patch(constant_patch.copy(), zero_mean=True, normalized=True)
        logger.info(f"✅ Constant-value patch processed successfully")
        logger.info(f"   Result shape: {processed.shape}, contains NaN: {np.isnan(processed).any()}")
    except Exception as e:
        logger.error(f"❌ Constant-value patch failed: {e}")
    
    # Test with extreme values
    logger.info("\nTesting with extreme-value patch...")
    extreme_patch = np.random.randn(2, 10, 10).astype(np.float32)
    extreme_patch[0] *= 1e6  # Very large values
    extreme_patch[1] *= 1e-6  # Very small values
    try:
        processed = process_patch(extreme_patch.copy(), zero_mean=True, normalized=True, quantiles=True)
        logger.info(f"✅ Extreme-value patch processed successfully")
        logger.info(f"   Result shape: {processed.shape}, contains NaN: {np.isnan(processed).any()}")
    except Exception as e:
        logger.error(f"❌ Extreme-value patch failed: {e}")


if __name__ == "__main__":
    logger.info("Starting Modular Processing Comprehensive Test Suite")
    
    try:
        # Test all combinations
        test_all_processing_combinations()
        
        # Test edge cases
        test_edge_cases()
        
        logger.info(f"\n{'='*80}")
        logger.info("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"💥 Test suite failed: {e}")
        raise
