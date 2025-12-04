#!/usr/bin/env python3

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
"""
Generate example patch plots for each class and model ID.
Structure: data/example_patch_plots/{class}/{id}.png
Uses the same patch for each class across all models.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_completed_models() -> List[Tuple[str, Path]]:
    """Find all completed training models with their IDs."""
    training_output_dir = Path("data/training_output")
    
    if not training_output_dir.exists():
        logger.warning(f"Training output directory not found: {training_output_dir}")
        return []
    
    models = []
    for model_dir in training_output_dir.iterdir():
        if model_dir.is_dir():
            # Check if model has results
            results_file = model_dir / "latest_results.txt"
            if results_file.exists():
                # Extract ID from directory name (remove hash suffix)
                dir_name = model_dir.name
                if '_' in dir_name:
                    model_id = '_'.join(dir_name.split('_')[:-1])  # Remove last part (hash)
                else:
                    model_id = dir_name
                models.append((model_id, model_dir))
                logger.info(f"Found model: {model_id}")
    
    logger.info(f"Found {len(models)} completed models")
    return models

def load_patch_data() -> Optional[Dict]:
    """Load patch data from the first available model."""
    # Try to find patch data from any model
    training_output_dir = Path("data/training_output")
    
    for model_dir in training_output_dir.iterdir():
        if model_dir.is_dir():
            # Look for saved patch data or logs
            logs_dir = model_dir / "logs"
            if logs_dir.exists():
                # Check if there are any numpy files with patch data
                for log_file in logs_dir.glob("*.npy"):
                    try:
                        data = np.load(log_file, allow_pickle=True)
                        if hasattr(data, 'item') and isinstance(data.item(), dict):
                            return data.item()
                    except:
                        continue
    
    # If no saved patch data, we'll need to extract from model training
    logger.warning("No patch data found in logs. Will need to extract from training data.")
    return None

def extract_sample_patches_per_class(classes: List[int] = [1, 4, 6, 12]) -> Dict[int, np.ndarray]:
    """Extract one sample patch per class for visualization."""
    
    # Import the necessary modules
    try:
        from src.lc_speckle_analysis.data_config import TrainingDataConfig
        from src.lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return {}
    
    # Load base configuration
    try:
        config_path = Path("configs/config_base.conf")
        config = TrainingDataConfig.from_file(config_path)
        logger.info("Loaded base configuration for patch extraction")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}
    
    # Initialize patch yielder
    try:
        patch_yielder = PatchYielder(config)
        logger.info("Initialized patch yielder")
    except Exception as e:
        logger.error(f"Failed to initialize patch yielder: {e}")
        return {}
    
    # Extract sample patches
    sample_patches = {}
    
    try:
        # Get training data generator that yields raw patch objects
        from src.lc_speckle_analysis.patch_data_extraction import PatchYielder
        
        # Create a new yielder to get raw patch objects
        raw_yielder = PatchYielder(config)
        
        # Get raw patches with metadata
        raw_patches = []
        for image_tuple_idx, image_tuple in enumerate(raw_yielder.image_tuples):
            # Extract patches from this image
            batch_generator = raw_yielder._extract_patches_from_image_tuple(image_tuple, DataMode.TRAIN)
            
            for batch in batch_generator:
                raw_patches.extend(batch)
                
                # Stop after finding enough patches
                if len(raw_patches) > 100:
                    break
            
            if len(raw_patches) > 100:
                break
        
        # Extract one patch per class from raw patches
        patches_found = {cls: False for cls in classes}
        
        for patch_obj in raw_patches:
            class_id = patch_obj.class_id
            
            if class_id in classes and not patches_found[class_id]:
                # Store both the processed array and the raw object
                sample_patches[class_id] = patch_obj.patch  # The numpy array
                patches_found[class_id] = True
                
                # Save the raw object to a separate file for metadata
                patch_obj_file = cache_dir / f"sample_patch_obj_class_{class_id}.pkl"
                with open(patch_obj_file, 'wb') as f:
                    pickle.dump(patch_obj, f)
                
                logger.info(f"Saved sample patch for class {class_id}")
                
                # Stop if we have all classes
                if all(patches_found.values()):
                    break
    
    except Exception as e:
        logger.error(f"Failed to extract patches: {e}")
        return {}
    
    # Save sample patches for reuse
    save_dir = Path("data/sample_patches")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for class_id, patch in sample_patches.items():
        save_path = save_dir / f"class_{class_id}_sample.npy"
        np.save(save_path, patch)
        logger.info(f"Saved sample patch for class {class_id}")
    
    return sample_patches

def load_or_extract_sample_patches() -> Dict[int, np.ndarray]:
    """Load existing sample patches from cached training data."""
    
    # Look for existing cached patch files
    cache_dir = Path("data/cache/patches/train")
    
    if not cache_dir.exists():
        logger.error(f"Training cache directory not found: {cache_dir}")
        return {}
    
    # Find pkl files in cache
    pkl_files = list(cache_dir.glob("*.pkl"))
    if not pkl_files:
        logger.error(f"No cached patch files found in {cache_dir}")
        return {}
    
    logger.info(f"Found {len(pkl_files)} cached patch files")
    
    # Target classes
    classes = [1, 4, 6, 12]
    sample_patches = {}
    
    # Load patches from first available cache file
    pkl_file = pkl_files[0]
    logger.info(f"Loading patches from: {pkl_file}")
    
    try:
        with open(pkl_file, 'rb') as f:
            cached_patches = pickle.load(f)
        
        logger.info(f"Loaded {len(cached_patches)} cached patches")
        
        # Extract one patch per class
        patches_found = {cls: False for cls in classes}
        
        for patch_obj in cached_patches:
            class_id = patch_obj.class_id
            
            if class_id in classes and not patches_found[class_id]:
                # Store the full patch object for metadata access
                sample_patches[class_id] = patch_obj
                patches_found[class_id] = True
                logger.info(f"Found sample patch for class {class_id}")
                
                # Stop if we have all classes
                if all(patches_found.values()):
                    break
        
        logger.info(f"Successfully extracted patches for classes: {list(sample_patches.keys())}")
        return sample_patches
        
    except Exception as e:
        logger.error(f"Failed to load cached patches: {e}")
        return {}

def get_model_config_from_name(model_id: str) -> Dict[str, any]:
    """Extract modular processing configuration from model name."""
    # Parse model name components
    parts = model_id.split('_')
    
    config = {
        'shuffled': False,
        'zero_mean': False,
        'normalized': False,
        'quantiles': False,
        'aggregation': None
    }
    
    # Parse configuration from model name parts
    for part in parts:
        if part == 'shuffled':
            config['shuffled'] = True
        elif part == 'zeromean':
            config['zero_mean'] = True
        elif part == 'normalized':
            config['normalized'] = True
        elif part == 'quantiles':
            config['quantiles'] = True
        elif part in ['std', 'mean', 'stdandmean']:
            config['aggregation'] = part
    
    return config

def create_patch_plot(patch_obj, class_id: int, model_id: str, output_path: Path):
    """Create a patch visualization plot with location hash in filename."""
    
    # Generate location hash from patch metadata
    import hashlib
    
    # Create location identifier from patch metadata
    location_data = {
        'bounds': patch_obj.bounds,
        'crs': str(patch_obj.crs),
        'date': patch_obj.date,
        'orbit': patch_obj.orbit,
        'data_mode': patch_obj.data_mode.value,
        'class_id': patch_obj.class_id
    }
    
    # Create stable hash from location data
    location_str = f"{location_data['bounds']}_{location_data['crs']}_{location_data['date']}_{location_data['orbit']}_{location_data['data_mode']}_{location_data['class_id']}"
    location_hash = hashlib.md5(location_str.encode()).hexdigest()[:8]
    
    # Get model-specific configuration
    model_config = get_model_config_from_name(model_id)
    
    # Get processed patch data using the model's specific configuration
    patch_data = patch_obj.get_data(
        shuffled=model_config['shuffled'],
        zero_mean=model_config['zero_mean'],
        normalized=model_config['normalized'],
        quantiles=model_config['quantiles'],
        aggregation=model_config['aggregation']
    )
    
    # Modify output path to include location hash
    output_path_with_hash = output_path.parent / f"{model_id}_{location_hash}.png"
    
    # Handle aggregated data (1D feature vector) vs patch data (3D array)
    if model_config['aggregation'] is not None:
        # For aggregated data, create a simple bar plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create bar plot
        bars = ax.bar(range(len(patch_data)), patch_data)
        ax.set_title(f'Aggregated Features - Class {class_id} - Model {model_id}')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Value')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars (only for first 10 to avoid clutter)
        for i, (bar, value) in enumerate(zip(bars[:10], patch_data[:10])):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01 * max(patch_data),
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    else:
        # Assert consistent (H, W, C) format
        assert patch_data.ndim == 3, f"Expected 3D patch data, got {patch_data.ndim}D with shape {patch_data.shape}"
        assert patch_data.shape[-1] == 2, f"Expected 2 channels (VV, VH), got {patch_data.shape[-1]} channels in shape {patch_data.shape}"
        
        # Extract channels from (H, W, C) format
        vv_channel = patch_data[:, :, 0]
        vh_channel = patch_data[:, :, 1]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Plot VV channel
        im1 = axes[0].imshow(vv_channel, cmap='gray', aspect='equal')
        axes[0].set_title('VV Polarization')
        axes[0].set_xlabel('Pixel X')
        axes[0].set_ylabel('Pixel Y')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Plot VH channel
        im2 = axes[1].imshow(vh_channel, cmap='gray', aspect='equal')
        axes[1].set_title('VH Polarization')
        axes[1].set_xlabel('Pixel X')
        axes[1].set_ylabel('Pixel Y')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Plot RGB composite (VV as red, VH as green)
        # Normalize channels for RGB display
        if vv_channel.max() > vv_channel.min() and vh_channel.max() > vh_channel.min():
            vv_norm = (vv_channel - vv_channel.min()) / (vv_channel.max() - vv_channel.min())
            vh_norm = (vh_channel - vh_channel.min()) / (vh_channel.max() - vh_channel.min())
        else:
            vv_norm = vv_channel
            vh_norm = vh_channel
        
        rgb_composite = np.zeros((*vv_channel.shape, 3))
        rgb_composite[:, :, 0] = vv_norm  # Red channel
        rgb_composite[:, :, 1] = vh_norm  # Green channel
        rgb_composite[:, :, 2] = 0        # Blue channel
        
        axes[2].imshow(rgb_composite, aspect='equal')
        axes[2].set_title('RGB Composite (VV+VH)')
        axes[2].set_xlabel('Pixel X')
        axes[2].set_ylabel('Pixel Y')
    
    # Add overall title with location and processing info
    processing_info = f"Shuffled: {model_config['shuffled']}, ZeroMean: {model_config['zero_mean']}, Normalized: {model_config['normalized']}, Quantiles: {model_config['quantiles']}, Aggregation: {model_config['aggregation']}"
    fig.suptitle(f'SAR Patch - Class {class_id} - Model {model_id}\nLocation: {location_hash} | {processing_info}', 
                 fontsize=9, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot with location hash in filename
    output_path_with_hash.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_with_hash, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created patch plot: {output_path_with_hash}")
    
    return location_hash

def main():
    """Main function to create example patch plots."""
    logger.info("Starting example patch plot generation...")
    
    # Find all completed models
    models = find_completed_models()
    
    if not models:
        logger.error("No completed models found")
        return
    
    # Load or extract sample patches
    sample_patches = load_or_extract_sample_patches()
    
    if not sample_patches:
        logger.error("Failed to obtain sample patches")
        return
    
    classes = sorted(sample_patches.keys())
    logger.info(f"Creating plots for classes: {classes}")
    
    # Create output directory structure
    output_base = Path("data/example_patch_plots")
    
    # Generate plots for each class and model
    total_plots = len(classes) * len(models)
    plot_count = 0
    
    for class_id in classes:
        class_dir = output_base / str(class_id)
        class_dir.mkdir(parents=True, exist_ok=True)
        
        patch_obj = sample_patches[class_id]  # Use the patch object from loaded patches
        
        for model_id, model_path in models:
            # Create plot filename
            output_file = class_dir / f"{model_id}.png"
            
            # Create the patch plot
            create_patch_plot(patch_obj, class_id, model_id, output_file)
            
            plot_count += 1
            logger.info(f"Progress: {plot_count}/{total_plots} plots created")
    
    logger.info(f"Example patch plot generation completed!")
    logger.info(f"Created {total_plots} plots in {output_base}")
    
    # Show summary
    for class_id in classes:
        class_dir = output_base / str(class_id)
        plot_files = list(class_dir.glob("*.png"))
        logger.info(f"Class {class_id}: {len(plot_files)} plots created")

if __name__ == "__main__":
    main()
