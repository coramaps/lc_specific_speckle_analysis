#!/usr/bin/env python3
"""
Enhanced patch visualization script for all modus options.

Creates comprehensive visualizations for:
- Spatial processing methods (Conv2D_N2): raw, zero_mean, quantiles, spatial_shuffle_0mean, spatial_shuffle
- Statistical processing methods (LinearStatsNet): std, meanandstd, mean

Generates:
1. TIFF files for spatial methods
2. PNG overview plots for spatial methods  
3. Mini feature visualizations for statistical methods
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.preprocessing import QuantileTransformer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lc_speckle_analysis import get_training_config
from src.lc_speckle_analysis.patch_yielder import PatchYielder, DataMode

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
        modus: Processing mode ('raw', 'zero_mean', 'quantiles', 'spatial_shuffle', 'spatial_shuffle_0mean')
        
    Returns:
        Processed patch data
    """
    if modus == "raw":
        return patch_data.copy()
        
    elif modus == "zero_mean" or modus == "data_with_zero_mean":
        # Subtract per-channel mean
        processed_data = patch_data.copy()
        for channel in range(patch_data.shape[2]):
            channel_mean = np.mean(processed_data[:, :, channel])
            processed_data[:, :, channel] -= channel_mean
        return processed_data
        
    elif modus == "quantiles":
        # Quantile transformation
        processed_data = patch_data.copy()
        height, width, channels = processed_data.shape
        
        for channel in range(channels):
            channel_data = processed_data[:, :, channel]
            flat_data = channel_data.flatten().reshape(-1, 1)
            
            # Use quantile transformer
            qt = QuantileTransformer(output_distribution='uniform', random_state=42)
            transformed_flat = qt.fit_transform(flat_data)
            processed_data[:, :, channel] = transformed_flat.reshape(height, width)
            
        return processed_data
        
    elif modus == "spatial_shuffle":
        # Shuffle pixels spatially while preserving spectral relationships
        processed_data = patch_data.copy()
        height, width, channels = processed_data.shape
        
        # Create consistent permutation for all channels
        np.random.seed(42)  # Fixed seed for reproducibility
        n_pixels = height * width
        perm_indices = np.random.permutation(n_pixels)
        
        # Apply same shuffling to all channels
        for channel_idx in range(channels):
            channel_data = processed_data[:, :, channel_idx]
            flat_data = channel_data.flatten()
            shuffled_data = flat_data[perm_indices]
            processed_data[:, :, channel_idx] = shuffled_data.reshape(height, width)
            
        return processed_data
        
    elif modus == "spatial_shuffle_0mean":
        # Apply spatial shuffling then zero-mean normalization
        shuffled_data = apply_modus_processing(patch_data, "spatial_shuffle")
        return apply_modus_processing(shuffled_data, "zero_mean")
        
    else:
        raise ValueError(f"Unknown modus: {modus}")


def extract_statistical_features(patch_data: np.ndarray, feature_type: str) -> np.ndarray:
    """
    Extract statistical features from patch data.
    
    Args:
        patch_data: Raw patch data with shape (height, width, channels)
        feature_type: 'mean', 'std', or 'meanandstd'
        
    Returns:
        Feature vector
    """
    vv_channel = patch_data[:, :, 0]
    vh_channel = patch_data[:, :, 1]
    
    if feature_type == "mean":
        return np.array([np.mean(vv_channel), np.mean(vh_channel)])
    elif feature_type == "std":
        return np.array([np.std(vv_channel), np.std(vh_channel)])
    elif feature_type == "meanandstd":
        return np.array([
            np.mean(vv_channel), np.std(vv_channel),
            np.mean(vh_channel), np.std(vh_channel)
        ])
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


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


def create_spatial_overview_plot(patches_dict: Dict[str, np.ndarray], patch_idx: int, 
                               output_dir: Path, class_label: int):
    """
    Create overview plot for spatial processing methods (Conv2D_N2).
    
    Args:
        patches_dict: Dictionary with modus as key and processed patch as value
        patch_idx: Index of the patch for naming
        output_dir: Output directory
        class_label: Class label for the patch
    """
    spatial_methods = ["raw", "zero_mean", "quantiles", "spatial_shuffle_0mean", "spatial_shuffle"]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Patch {patch_idx+1} (Class {class_label}) - Spatial Processing Methods (Conv2D_N2)', 
                fontsize=16, fontweight='bold')
    
    for col, modus in enumerate(spatial_methods):
        if modus not in patches_dict:
            continue
            
        patch_data = patches_dict[modus]
        
        # VV channel (top row)
        vv_data = patch_data[:, :, 0]
        im1 = axes[0, col].imshow(vv_data, cmap='viridis')
        axes[0, col].set_title(f'{modus.replace("_", " ").title()}\nVV Channel', fontweight='bold')
        axes[0, col].axis('off')
        plt.colorbar(im1, ax=axes[0, col], shrink=0.8)
        
        # VH channel (bottom row)
        vh_data = patch_data[:, :, 1]
        im2 = axes[1, col].imshow(vh_data, cmap='plasma')
        axes[1, col].set_title(f'{modus.replace("_", " ").title()}\nVH Channel', fontweight='bold')
        axes[1, col].axis('off')
        plt.colorbar(im2, ax=axes[1, col], shrink=0.8)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plot_path = output_dir / f"spatial_overview_patch_{patch_idx+1}_class_{class_label}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved spatial overview plot: {plot_path}")


def create_statistical_mini_plot(features_dict: Dict[str, np.ndarray], patch_idx: int,
                               output_dir: Path, class_label: int):
    """
    Create mini visualization for statistical features (LinearStatsNet).
    
    Args:
        features_dict: Dictionary with feature type as key and feature vector as value
        patch_idx: Index of the patch for naming
        output_dir: Output directory
        class_label: Class label for the patch
    """
    feature_types = ["std", "meanandstd", "mean"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Patch {patch_idx+1} (Class {class_label}) - Statistical Features (LinearStatsNet)', 
                fontsize=14, fontweight='bold')
    
    for col, feat_type in enumerate(feature_types):
        if feat_type not in features_dict:
            continue
            
        features = features_dict[feat_type]
        
        if feat_type == "mean":
            # 2 features: mean_VV, mean_VH
            data_2d = features.reshape(1, 2)
            labels = ['VV_mean', 'VH_mean']
        elif feat_type == "std":
            # 2 features: std_VV, std_VH  
            data_2d = features.reshape(1, 2)
            labels = ['VV_std', 'VH_std']
        elif feat_type == "meanandstd":
            # 4 features: mean_VV, std_VV, mean_VH, std_VH
            data_2d = features.reshape(2, 2)
            labels = ['VV_mean', 'VV_std', 'VH_mean', 'VH_std']
        
        # Create mini image visualization - use viridis to match spatial plots
        im = axes[col].imshow(data_2d, cmap='viridis', aspect='auto')
        axes[col].set_title(f'{feat_type.replace("and", " & ").upper()}\n({len(features)} features)', 
                           fontweight='bold')
        
        # Add feature labels
        if feat_type in ["mean", "std"]:
            axes[col].set_xticks([0, 1])
            axes[col].set_xticklabels(['VV', 'VH'])
            axes[col].set_yticks([])
        else:  # meanandstd
            axes[col].set_xticks([0, 1])
            axes[col].set_xticklabels(['Mean', 'Std'])
            axes[col].set_yticks([0, 1])
            axes[col].set_yticklabels(['VV', 'VH'])
        
        # Add feature values as text
        for i in range(data_2d.shape[0]):
            for j in range(data_2d.shape[1]):
                value = data_2d[i, j]
                axes[col].text(j, i, f'{value:.3f}', ha='center', va='center', 
                             fontweight='bold', fontsize=10,
                             color='white' if abs(value) > np.abs(data_2d).max()/2 else 'black')
        
        plt.colorbar(im, ax=axes[col], shrink=0.8)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plot_path = output_dir / f"statistical_features_patch_{patch_idx+1}_class_{class_label}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved statistical features plot: {plot_path}")


def main():
    """Main function to create comprehensive patch visualizations."""
    logger.info("Starting comprehensive patch visualization...")
    
    # Load configuration
    config = get_training_config()
    logger.info(f"Loaded configuration with modus: {config.modus}")
    
    # Create output directory
    output_dir = project_root / "results" / "comprehensive_patch_visualizations"
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
    
    logger.info(f"Selected {n_samples} random patches for comprehensive visualization")
    
    # Define processing methods
    spatial_methods = ["raw", "zero_mean", "quantiles", "spatial_shuffle_0mean", "spatial_shuffle"]
    statistical_methods = ["std", "meanandstd", "mean"]
    
    # Process each selected patch
    for i, patch_idx in enumerate(selected_indices):
        patch_data = sample_patches[patch_idx]
        class_label = sample_labels[patch_idx]
        
        logger.info(f"\nProcessing patch {i+1}/{n_samples} (class {class_label})...")
        
        # Process spatial methods
        processed_spatial = {}
        for modus in spatial_methods:
            logger.info(f"  Processing spatial method: {modus}")
            processed_data = apply_modus_processing(patch_data, modus)
            processed_spatial[modus] = processed_data
            
            # Save as TIFF
            tiff_dir = output_dir / "tiff_files" / f"patch_{i+1}_class_{class_label}"
            tiff_path = tiff_dir / f"{modus}.tif"
            save_patch_as_tiff(processed_data, tiff_path)
        
        # Create spatial overview plot
        create_spatial_overview_plot(processed_spatial, i, output_dir / "spatial_overviews", class_label)
        
        # Extract statistical features
        statistical_features = {}
        for feat_type in statistical_methods:
            logger.info(f"  Extracting statistical features: {feat_type}")
            features = extract_statistical_features(patch_data, feat_type)
            statistical_features[feat_type] = features
        
        # Create statistical mini visualization
        create_statistical_mini_plot(statistical_features, i, output_dir / "statistical_mini_plots", class_label)
    
    # Create comprehensive summary info
    summary_path = output_dir / "visualization_info.txt"
    with open(summary_path, 'w') as f:
        f.write("Comprehensive Patch Visualization Information\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {Path(__file__).name}\n")
        f.write(f"Configuration modus: {config.modus}\n")
        f.write(f"Patch size: {config.neural_network.patch_size}x{config.neural_network.patch_size}\n")
        f.write(f"Classes: {config.classes}\n")
        f.write(f"Total patches sampled: {n_samples}\n\n")
        
        f.write("Spatial Processing Methods (Conv2D_N2):\n")
        f.write("- raw: Original pixel values (no preprocessing)\n")
        f.write("- zero_mean: Per-channel mean subtraction\n")
        f.write("- quantiles: Quantile transformation (spatial structure preserved)\n")
        f.write("- spatial_shuffle_0mean: Spatial shuffling + zero-mean normalization\n")
        f.write("- spatial_shuffle: Pixel shuffling (spectral relationships preserved)\n\n")
        
        f.write("Statistical Processing Methods (LinearStatsNet):\n")
        f.write("- std: Standard deviation features (2 features: std_VV, std_VH)\n")
        f.write("- meanandstd: Combined mean and std features (4 features: mean_VV, std_VV, mean_VH, std_VH)\n")
        f.write("- mean: Mean features (2 features: mean_VV, mean_VH)\n\n")
        
        f.write("Generated Files:\n")
        f.write("- tiff_files/: GeoTIFF files for spatial processing methods\n")
        f.write("- spatial_overviews/: PNG overview plots for spatial methods\n")
        f.write("- statistical_mini_plots/: Mini visualizations for statistical features\n\n")
        
        f.write("Selected Patches:\n")
        for i, patch_idx in enumerate(selected_indices):
            class_label = sample_labels[patch_idx]
            f.write(f"Patch {i+1}: Original index {patch_idx}, Class {class_label}\n")
    
    logger.info(f"Summary info saved to: {summary_path}")
    logger.info("✅ Comprehensive patch visualization completed successfully!")
    
    # Print summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE PATCH VISUALIZATIONS CREATED")
    print(f"{'='*80}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Patches sampled: {n_samples}")
    print(f"🔧 Spatial methods: {len(spatial_methods)} (Conv2D_N2)")
    print(f"📊 Statistical methods: {len(statistical_methods)} (LinearStatsNet)")
    print(f"📁 TIFF files: {output_dir}/tiff_files/")
    print(f"🖼️  Spatial overviews: {output_dir}/spatial_overviews/")
    print(f"🔢 Statistical mini plots: {output_dir}/statistical_mini_plots/")
    print(f"📄 Summary info: {summary_path}")


if __name__ == "__main__":
    main()
