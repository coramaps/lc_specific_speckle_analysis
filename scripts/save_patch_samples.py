#!/usr/bin/env python3
"""
Save sample patches processed with different modus options as TIFF files.

This script extracts 5 random patches and saves them processed with each modus:
- raw: Original pixel values
- data_with_zero_mean: Zero-mean normalized
- quantiles: Quantile transformed (spatial info only)
- spatial_shuffle: Spatially shuffled (spectral info only)

The same 5 patches are used for all modus options to enable direct comparison.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lc_speckle_analysis import get_training_config
from src.lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
from src.lc_speckle_analysis.train_model import PatchDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')


def apply_modus_processing(patch_data: np.ndarray, modus: str) -> np.ndarray:
    """
    Apply modus processing to patch data.
    
    Args:
        patch_data: Raw patch data with shape (height, width, channels)
        modus: Processing mode
        
    Returns:
        Processed patch data
    """
    assert patch_data.ndim == 3, f"Expected 3D patch data, got {patch_data.ndim}D with shape {patch_data.shape}"
    assert patch_data.shape[-1] == 2, f"Expected 2 channels (VV, VH), got {patch_data.shape[-1]} channels in shape {patch_data.shape}"
    
    processed_data = patch_data.copy()
    
    if modus == "raw":
        return processed_data
        
    elif modus == "data_with_zero_mean":
        # Apply zero-mean normalization per channel (VV and VH)
        for channel_idx in range(processed_data.shape[2]):
            channel_data = processed_data[:, :, channel_idx]
            channel_mean = np.mean(channel_data)
            processed_data[:, :, channel_idx] = channel_data - channel_mean
        return processed_data
        
    elif modus == "quantiles":
        # Transform each patch to quantiles (0.00, 0.01, ..., 1.00)
        from scipy.stats import rankdata
        
        for channel_idx in range(processed_data.shape[2]):
            channel_data = processed_data[:, :, channel_idx]
            flat_data = channel_data.flatten()
            
            # Use scipy.stats.rankdata to get ranks, then normalize to [0,1]
            ranks = rankdata(flat_data, method='average')  # Handles ties properly
            quantile_data = (ranks - 1) / (len(flat_data) - 1)  # Normalize to [0,1]
            
            processed_data[:, :, channel_idx] = quantile_data.reshape(channel_data.shape)
        return processed_data
        
    elif modus == "spatial_shuffle":
        # Shuffle pixels within each patch (same indices for VV and VH)
        height, width, channels = processed_data.shape
        n_pixels = height * width
        
        # Generate random permutation for pixel positions (deterministic per patch)
        patch_seed = int(np.sum(patch_data) * 1000) % (2**31)  # Convert to valid seed
        np.random.seed(patch_seed)
        perm_indices = np.random.permutation(n_pixels)
        
        # Apply same shuffling to all channels
        for channel_idx in range(channels):
            channel_data = processed_data[:, :, channel_idx]
            flat_data = channel_data.flatten()
            shuffled_data = flat_data[perm_indices]
            processed_data[:, :, channel_idx] = shuffled_data.reshape(height, width)
            
        return processed_data
        
    else:
        raise ValueError(f"Unknown modus: {modus}")


def save_patch_as_tiff(patch_data: np.ndarray, output_path: Path, 
                      patch_bounds: Tuple[float, float, float, float] = None,
                      crs: str = "EPSG:4326"):
    """
    Save patch data as GeoTIFF file.
    
    Args:
        patch_data: Patch data with shape (height, width, channels)
        output_path: Output TIFF file path
        patch_bounds: (minx, miny, maxx, maxy) bounds for georeferencing
        crs: Coordinate reference system
    """
    assert patch_data.ndim == 3, f"Expected 3D patch data, got {patch_data.ndim}D with shape {patch_data.shape}"
    assert patch_data.shape[-1] == 2, f"Expected 2 channels (VV, VH), got {patch_data.shape[-1]} channels in shape {patch_data.shape}"
    
    height, width, channels = patch_data.shape
    
    # Create transform from bounds (if provided) or use identity
    if patch_bounds:
        minx, miny, maxx, maxy = patch_bounds
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
    else:
        transform = rasterio.transform.from_bounds(0, 0, width, height, width, height)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write TIFF file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=channels,
        dtype=patch_data.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        for i in range(channels):
            dst.write(patch_data[:, :, i], i + 1)
    
    logger.info(f"Saved patch to: {output_path}")


def create_visual_comparison(patches_dict: dict, patch_idx: int, output_dir: Path, class_label: int):
    """
    Create visual comparison plot for a single patch across all modus options.
    
    Args:
        patches_dict: Dictionary with modus as key and processed patch as value
        patch_idx: Index of the patch for naming
        output_dir: Output directory
        class_label: Class label for the patch
    """
    modus_list = ["raw", "data_with_zero_mean", "quantiles", "spatial_shuffle"]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Patch {patch_idx+1} (Class {class_label}) - Modus Comparison', fontsize=14)
    
    for col, modus in enumerate(modus_list):
        patch_data = patches_dict[modus]
        
        # VV channel (top row)
        vv_data = patch_data[:, :, 0]
        im1 = axes[0, col].imshow(vv_data, cmap='viridis')
        axes[0, col].set_title(f'{modus}\nVV Channel')
        axes[0, col].axis('off')
        plt.colorbar(im1, ax=axes[0, col], shrink=0.8)
        
        # VH channel (bottom row)
        vh_data = patch_data[:, :, 1]
        im2 = axes[1, col].imshow(vh_data, cmap='plasma')
        axes[1, col].set_title(f'{modus}\nVH Channel')
        axes[1, col].axis('off')
        plt.colorbar(im2, ax=axes[1, col], shrink=0.8)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plot_path = output_dir / f"patch_{patch_idx+1}_class_{class_label}_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plot: {plot_path}")


def main():
    """Main function to save sample patches."""
    logger.info("Starting patch sample extraction...")
    
    # Load configuration
    config = get_training_config()
    logger.info(f"Loaded configuration with modus: {config.modus}")
    
    # Create output directory
    output_dir = project_root / "results" / "patch_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Create patch yielder
    patch_yielder = PatchYielder(config)
    
    # Extract sample patches from training set
    logger.info("Extracting sample patches from training set...")
    sample_patches = []
    sample_labels = []
    
    # Use a fixed seed for reproducible patch selection
    np.random.seed(42)
    
    # Collect patches
    patch_count = 0
    for patch_data, class_id in patch_yielder.yield_patch(DataMode.TRAIN):
        sample_patches.append(patch_data)
        sample_labels.append(class_id)
        patch_count += 1
        
        # Stop after collecting enough patches to choose from
        if patch_count >= 50:
            break
    
    logger.info(f"Collected {len(sample_patches)} patches")
    
    # Select 5 random patches
    n_samples = min(5, len(sample_patches))
    selected_indices = np.random.choice(len(sample_patches), size=n_samples, replace=False)
    
    logger.info(f"Selected {n_samples} random patches for modus comparison")
    
    # Modus options to test
    modus_options = ["raw", "data_with_zero_mean", "quantiles", "spatial_shuffle"]
    
    # Process and save each selected patch with all modus options
    for i, patch_idx in enumerate(selected_indices):
        patch_data = sample_patches[patch_idx]
        class_label = sample_labels[patch_idx]
        
        logger.info(f"\nProcessing patch {i+1}/5 (class {class_label})...")
        
        # Store processed patches for comparison
        processed_patches = {}
        
        # Process patch with each modus
        for modus in modus_options:
            processed_data = apply_modus_processing(patch_data, modus)
            processed_patches[modus] = processed_data
            
            # Save as TIFF
            tiff_dir = output_dir / "tiff_files" / f"patch_{i+1}_class_{class_label}"
            tiff_path = tiff_dir / f"{modus}.tif"
            save_patch_as_tiff(processed_data, tiff_path)
            
            # Log data statistics
            logger.info(f"  {modus}: VV range [{processed_data[:,:,0].min():.3f}, {processed_data[:,:,0].max():.3f}], "
                       f"VH range [{processed_data[:,:,1].min():.3f}, {processed_data[:,:,1].max():.3f}]")
        
        # Create visual comparison
        create_visual_comparison(processed_patches, i, output_dir / "comparisons", class_label)
    
    # Create summary info file
    summary_path = output_dir / "sample_info.txt"
    with open(summary_path, 'w') as f:
        f.write("Patch Sample Information\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {Path(__file__).name}\n")
        f.write(f"Configuration modus: {config.modus}\n")
        f.write(f"Patch size: {config.neural_network.patch_size}x{config.neural_network.patch_size}\n")
        f.write(f"Classes: {config.classes}\n")
        f.write(f"Total patches sampled: {n_samples}\n\n")
        
        f.write("Modus Descriptions:\n")
        f.write("- raw: Original pixel values (no preprocessing)\n")
        f.write("- data_with_zero_mean: Per-channel mean subtraction\n")
        f.write("- quantiles: Quantile transformation (spatial info only)\n")
        f.write("- spatial_shuffle: Pixel shuffling (spectral info only)\n\n")
        
        f.write("Selected Patches:\n")
        for i, patch_idx in enumerate(selected_indices):
            class_label = sample_labels[patch_idx]
            f.write(f"Patch {i+1}: Original index {patch_idx}, Class {class_label}\n")
    
    logger.info(f"Sample info saved to: {summary_path}")
    logger.info("✅ Patch sample extraction completed successfully!")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PATCH SAMPLES CREATED")
    print(f"{'='*60}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Patches sampled: {n_samples}")
    print(f"🔧 Modus options: {len(modus_options)}")
    print(f"📁 TIFF files: {output_dir}/tiff_files/")
    print(f"🖼️  Comparison plots: {output_dir}/comparisons/")
    print(f"📄 Summary info: {summary_path}")


if __name__ == "__main__":
    main()
