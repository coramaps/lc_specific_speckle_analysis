#!/usr/bin/env python3
"""
Create bar plots comparing Overall Accuracy and mean F1 scores across all modus configurations.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_metrics_from_results():
    """Extract OA and mean F1 metrics from all training result directories."""
    
    results = []
    data_dir = Path("/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output")
    
    # Based on the terminal output, we need to find directories for these specific run IDs:
    # The terminal shows these exact run completions from today's training
    target_runs = [
        "normal_datawithzeromean_single_20220611_conv2d",
        "normal_raw_single_20220611_conv2d", 
        "normal_quantiles_single_20220611_conv2d",
        "normal_spatialshuffle_single_20220611_conv2d",
        "normal_datawithzeromean_single_20220611_conv2d_n2",
        "normal_raw_single_20220611_conv2d_n2",
        "normal_quantiles_single_20220611_conv2d_n2",
        "normal_spatialshuffle_single_20220611_conv2d_n2"
    ]
    
    # Get all directories and their creation times
    import time
    today_start = time.time() - 24*3600  # 24 hours ago
    
    for target_run in target_runs:
        # Extract modus and architecture from run name
        if "datawithzeromean" in target_run:
            modus = "data_with_zero_mean"
        elif "quantiles" in target_run:
            modus = "quantiles"
        elif "spatialshuffle" in target_run:
            modus = "spatial_shuffle"
        elif "raw" in target_run:
            modus = "raw"
        else:
            modus = "unknown"
            
        if "conv2d_n2" in target_run:
            arch = "TestConv2D_N2"
        elif "conv2d" in target_run:
            arch = "TestConv2D"
        else:
            arch = "unknown"
        
        # Find directories that match this run pattern and were created today
        pattern = f"run_{target_run}_*"
        matching_dirs = list(data_dir.glob(pattern))
        
        if not matching_dirs:
            logger.warning(f"No directory found for {target_run}")
            continue
            
        # Filter for directories created today and sort by most recent
        recent_dirs = [d for d in matching_dirs if d.stat().st_ctime > today_start]
        if not recent_dirs:
            logger.warning(f"No recent directories for {target_run}, using most recent available")
            recent_dirs = matching_dirs
            
        recent_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        run_dir = recent_dirs[0]
        
        logger.info(f"Processing {target_run} from {run_dir}")
        
        # Find training summary JSON files with today's timestamp pattern (20251202_*)
        json_files = list(run_dir.glob("training_summary_20251202_*.json"))
        if not json_files:
            # Fallback to any training summary
            json_files = list(run_dir.glob("training_summary_*.json"))
            
        if not json_files:
            logger.warning(f"No training summary found in {run_dir}")
            continue
            
        # Use the most recent JSON file
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        json_file = json_files[0]
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Extract metrics
            oa = data['test_results']['test_accuracy']
            mean_f1 = data['test_results']['macro_avg']['f1-score']
            
            results.append({
                'run_id': target_run,
                'modus': modus,
                'architecture': arch,
                'overall_accuracy': oa,
                'mean_f1': mean_f1,
                'directory': str(run_dir),
                'json_file': str(json_file)
            })
            
            logger.info(f"  OA: {oa:.4f}, Mean F1: {mean_f1:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            
    return results

def create_bar_plots(results):
    """Create bar plots for Overall Accuracy and mean F1 scores."""
    
    if not results:
        logger.error("No results to plot")
        return
        
    # Sort results by modus and architecture for consistent plotting
    modus_order = ['data_with_zero_mean', 'raw', 'quantiles', 'spatial_shuffle']
    arch_order = ['TestConv2D', 'TestConv2D_N2']
    
    # Create labels for x-axis
    labels = []
    oa_values = []
    f1_values = []
    colors = []
    
    # Define colors for each modus
    modus_colors = {
        'data_with_zero_mean': '#1f77b4',  # blue
        'raw': '#ff7f0e',                  # orange
        'quantiles': '#2ca02c',            # green
        'spatial_shuffle': '#d62728'       # red
    }
    
    # Organize results by architecture and modus
    for arch in arch_order:
        for modus in modus_order:
            # Find matching result
            matching = [r for r in results if r['architecture'] == arch and r['modus'] == modus]
            if matching:
                result = matching[0]
                labels.append(f"{modus}\n({arch})")
                oa_values.append(result['overall_accuracy'])
                f1_values.append(result['mean_f1'])
                colors.append(modus_colors[modus])
            else:
                logger.warning(f"Missing result for {arch} + {modus}")
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall Accuracy plot
    bars1 = ax1.bar(labels, oa_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_title('Overall Accuracy by Modus and Architecture', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Overall Accuracy', fontsize=12)
    ax1.set_xlabel('Configuration (Modus + Architecture)', fontsize=12)
    ax1.tick_labels = labels
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(oa_values) * 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars1, oa_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Mean F1 Score plot
    bars2 = ax2.bar(labels, f1_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_title('Mean F1 Score by Modus and Architecture', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean F1 Score', fontsize=12)
    ax2.set_xlabel('Configuration (Modus + Architecture)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(f1_values) * 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars2, f1_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Create custom legend for modus
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, edgecolor='black') 
                      for color in modus_colors.values()]
    legend_labels = [modus.replace('_', ' ').title() for modus in modus_colors.keys()]
    
    fig.legend(legend_elements, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
              ncol=4, title='Data Processing Modus', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the plot
    results_dir = Path("/home/davideidmann/code/lc_specific_speckle_analysis/results")
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "modus_comparison_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Bar plots saved to: {output_file}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY BY MODUS")
    print("="*80)
    
    # Group by modus
    modus_stats = {}
    for result in results:
        modus = result['modus']
        if modus not in modus_stats:
            modus_stats[modus] = {'oa': [], 'f1': []}
        modus_stats[modus]['oa'].append(result['overall_accuracy'])
        modus_stats[modus]['f1'].append(result['mean_f1'])
    
    for modus in modus_order:
        if modus in modus_stats:
            oa_mean = np.mean(modus_stats[modus]['oa'])
            f1_mean = np.mean(modus_stats[modus]['f1'])
            print(f"{modus:20s} | OA: {oa_mean:.4f} | F1: {f1_mean:.4f}")
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY BY ARCHITECTURE")
    print("="*80)
    
    # Group by architecture
    arch_stats = {}
    for result in results:
        arch = result['architecture']
        if arch not in arch_stats:
            arch_stats[arch] = {'oa': [], 'f1': []}
        arch_stats[arch]['oa'].append(result['overall_accuracy'])
        arch_stats[arch]['f1'].append(result['mean_f1'])
    
    for arch in arch_order:
        if arch in arch_stats:
            oa_mean = np.mean(arch_stats[arch]['oa'])
            f1_mean = np.mean(arch_stats[arch]['f1'])
            print(f"{arch:15s} | OA: {oa_mean:.4f} | F1: {f1_mean:.4f}")

def main():
    """Main function to run the analysis."""
    logger.info("Extracting metrics from training results...")
    results = extract_metrics_from_results()
    
    if not results:
        logger.error("No results found. Check if training outputs exist.")
        return
    
    logger.info(f"Found {len(results)} training results")
    
    # Create bar plots
    create_bar_plots(results)

if __name__ == "__main__":
    main()
