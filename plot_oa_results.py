#!/usr/bin/env python3
"""
Plot Overall Accuracy (OA) bar plot for all computed models.
Sorted by OA with unique_id (without hash) as x-axis labels.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_training_results() -> List[Path]:
    """Find all training output directories with results."""
    training_output_dir = Path("data/training_output")
    
    if not training_output_dir.exists():
        logger.warning(f"Training output directory not found: {training_output_dir}")
        return []
    
    result_dirs = []
    for model_dir in training_output_dir.iterdir():
        if model_dir.is_dir():
            # Look for latest_results.txt file
            results_file = model_dir / "latest_results.txt"
            if results_file.exists():
                result_dirs.append(model_dir)
                logger.info(f"Found results: {model_dir.name}")
    
    logger.info(f"Found {len(result_dirs)} models with results")
    return result_dirs

def extract_oa_from_results(results_dir: Path) -> Tuple[str, float, str, bool]:
    """Extract OA from results and get unique_id from config."""
    results_file = results_dir / "latest_results.txt"
    
    # Load test results from latest_results.txt
    try:
        with open(results_file, 'r') as f:
            content = f.read()
        
        # Parse Test Accuracy from the file
        oa = 0.0
        for line in content.split('\n'):
            if line.startswith('Test Accuracy:'):
                oa = float(line.split(':')[1].strip())
        
        # Find config file to get unique_id
        config_files = list(results_dir.glob("config_*.json"))
        if not config_files:
            logger.warning(f"No config file found in {results_dir}")
            return results_dir.name, oa, results_dir.name
        
        # Load config to get unique_id
        with open(config_files[0], 'r') as f:
            config = json.load(f)
        
        unique_name = config.get('unique_name', results_dir.name)
        
        # Remove hash suffix if present (format: name_hash_XXXX)
        if '_hash_' in unique_name:
            unique_id = unique_name.split('_hash_')[0]
        else:
            unique_id = unique_name
        
        # Extract aggregation info from config (nested in data_processing)
        data_processing = config.get('data_processing', {})
        aggregation = data_processing.get('aggregation', None)
        is_spatial = aggregation is None
        
        logger.info(f"Model {unique_id}: OA = {oa:.4f}, Spatial = {is_spatial}")
        return unique_id, oa, unique_name, is_spatial
        
    except Exception as e:
        logger.error(f"Error reading results from {results_dir}: {e}")
        return results_dir.name, 0.0, results_dir.name, True

def create_oa_barplot(model_data: List[Tuple[str, float, str, bool]], output_file: str = "oa_barplot.png"):
    """Create bar plot of OA values sorted by performance."""
    
    if not model_data:
        logger.error("No model data to plot")
        return
    
    # Sort by OA (descending)
    sorted_data = sorted(model_data, key=lambda x: x[1], reverse=True)
    
    unique_ids = [item[0] for item in sorted_data]
    oa_values = [item[1] for item in sorted_data]
    full_names = [item[2] for item in sorted_data]
    is_spatial = [item[3] for item in sorted_data]
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create bar plot with colors based on aggregation
    colors = ['steelblue' if spatial else 'darkorange' for spatial in is_spatial]
    bars = plt.bar(range(len(unique_ids)), oa_values, color=colors, alpha=0.7)
    
    # Customize plot
    plt.xlabel('Model Configuration (Unique ID)', fontsize=12)
    plt.ylabel('Overall Accuracy (OA)', fontsize=12)
    plt.title('Overall Accuracy by Model Configuration\n(Blue: Spatial Models, Orange: Statistical Models)', fontsize=14, fontweight='bold')
    
    # Add legend
    import matplotlib.patches as mpatches
    spatial_patch = mpatches.Patch(color='steelblue', alpha=0.7, label='Spatial Models (aggregation=None)')
    statistical_patch = mpatches.Patch(color='darkorange', alpha=0.7, label='Statistical Models (aggregation≠None)')
    plt.legend(handles=[spatial_patch, statistical_patch], loc='upper right')
    
    # Set x-axis labels with rotation
    plt.xticks(range(len(unique_ids)), unique_ids, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, oa, full_name) in enumerate(zip(bars, oa_values, full_names)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{oa:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Set y-axis limits with some padding
    if oa_values:
        y_min = max(0, min(oa_values) - 0.05)
        y_max = min(1, max(oa_values) + 0.05)
        plt.ylim(y_min, y_max)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    
    logger.info(f"Plot saved to: {output_file}")
    logger.info(f"Plot saved to: {output_file.replace('.png', '.pdf')}")
    
    # Show summary
    spatial_count = sum(is_spatial)
    statistical_count = len(is_spatial) - spatial_count
    logger.info(f"\nSummary of {len(model_data)} models:")
    logger.info(f"Spatial models (blue): {spatial_count}, Statistical models (orange): {statistical_count}")
    logger.info(f"Best OA: {max(oa_values):.4f} ({unique_ids[0]}) {'[Spatial]' if is_spatial[0] else '[Statistical]'}")
    logger.info(f"Worst OA: {min(oa_values):.4f} ({unique_ids[-1]}) {'[Spatial]' if is_spatial[-1] else '[Statistical]'}")
    logger.info(f"Mean OA: {np.mean(oa_values):.4f}")
    
    return output_file

def main():
    """Main function to create OA bar plot."""
    logger.info("Starting OA bar plot generation...")
    
    # Find all training results
    result_dirs = find_training_results()
    
    if not result_dirs:
        logger.error("No training results found")
        return
    
    # Extract OA data from all results
    model_data = []
    for results_dir in result_dirs:
        unique_id, oa, full_name, is_spatial = extract_oa_from_results(results_dir)
        model_data.append((unique_id, oa, full_name, is_spatial))
    
    # Create bar plot
    output_file = create_oa_barplot(model_data, "oa_results_barplot.png")
    
    logger.info("OA bar plot generation completed!")
    
    # Also save data to CSV for reference
    csv_file = "oa_results_summary.csv"
    with open(csv_file, 'w') as f:
        f.write("unique_id,overall_accuracy,full_name,is_spatial\n")
        for unique_id, oa, full_name, is_spatial in sorted(model_data, key=lambda x: x[1], reverse=True):
            f.write(f"{unique_id},{oa:.6f},{full_name},{is_spatial}\n")
    
    logger.info(f"Results summary saved to: {csv_file}")

if __name__ == "__main__":
    main()
