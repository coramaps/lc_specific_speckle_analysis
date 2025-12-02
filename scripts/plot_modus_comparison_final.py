#!/usr/bin/env python3
"""
Create bar plots comparing Overall Accuracy and mean F1 scores across all modus configurations.
Using the exact directories created today from the training runs.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_metrics_from_results():
    """Extract OA and mean F1 metrics from today's training result directories."""
    
    # Exact mapping of today's training directories (created 2025-12-02)
    training_dirs = {
        "data_with_zero_mean_TestConv2D": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_datawithzeromean_single_20220611_conv2d_5fda7aee",
        "data_with_zero_mean_TestConv2D_N2": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_datawithzeromean_single_20220611_conv2d_n2_c7877c44",
        "raw_TestConv2D": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_raw_single_20220611_conv2d_6a18e1d7",
        "raw_TestConv2D_N2": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_raw_single_20220611_conv2d_n2_0eb5f50f",
        "quantiles_TestConv2D": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_quantiles_single_20220611_conv2d_b1f70fec",
        "quantiles_TestConv2D_N2": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_quantiles_single_20220611_conv2d_n2_44f6d712",
        "spatial_shuffle_TestConv2D": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_spatialshuffle_single_20220611_conv2d_bbdcd624",
        "spatial_shuffle_TestConv2D_N2": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_spatialshuffle_single_20220611_conv2d_n2_3dbe63a0"
    }
    
    results = []
    
    for config_key, dir_path in training_dirs.items():
        run_dir = Path(dir_path)
        
        # Parse modus and architecture from config key
        parts = config_key.split('_')
        if len(parts) >= 3:
            modus = '_'.join(parts[:-1])  # Everything except architecture
            arch = parts[-1]
        else:
            logger.error(f"Could not parse config key: {config_key}")
            continue
            
        logger.info(f"Processing {config_key} from {run_dir}")
        
        if not run_dir.exists():
            logger.warning(f"Directory does not exist: {run_dir}")
            continue
            
        # Find training summary JSON files from today (20251202_*)
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
                'config_key': config_key,
                'modus': modus,
                'architecture': arch,
                'overall_accuracy': oa,
                'mean_f1': mean_f1,
                'directory': str(run_dir),
                'json_file': str(json_file)
            })
            
            logger.info(f"  Modus: {modus}, Arch: {arch}")
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
                arch_short = arch.replace('TestConv2D_N2', 'Conv2D_N2').replace('TestConv2D', 'Conv2D')
                modus_short = modus.replace('data_with_zero_mean', 'Zero Mean').replace('spatial_shuffle', 'Spatial Shuffle').replace('_', ' ').title()
                labels.append(f"{modus_short}\n({arch_short})")
                oa_values.append(result['overall_accuracy'])
                f1_values.append(result['mean_f1'])
                colors.append(modus_colors[modus])
            else:
                logger.warning(f"Missing result for {arch} + {modus}")
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Overall Accuracy plot
    bars1 = ax1.bar(range(len(labels)), oa_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Overall Accuracy by Modus and Architecture', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Overall Accuracy', fontsize=14)
    ax1.set_xlabel('Configuration (Modus + Architecture)', fontsize=14)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(oa_values) * 1.15)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, oa_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Mean F1 Score plot
    bars2 = ax2.bar(range(len(labels)), f1_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Mean F1 Score by Modus and Architecture', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Mean F1 Score', fontsize=14)
    ax2.set_xlabel('Configuration (Modus + Architecture)', fontsize=14)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(f1_values) * 1.15)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars2, f1_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Create custom legend for modus
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='black') 
                      for color in modus_colors.values()]
    legend_labels = ['Zero Mean', 'Raw', 'Quantiles', 'Spatial Shuffle']
    
    fig.legend(legend_elements, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=4, title='Data Processing Modus', fontsize=12, title_fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    # Save the plot
    results_dir = Path("/home/davideidmann/code/lc_specific_speckle_analysis/results")
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "modus_comparison_metrics_final.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Bar plots saved to: {output_file}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*90)
    print("PERFORMANCE SUMMARY BY MODUS")
    print("="*90)
    
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
            oa_std = np.std(modus_stats[modus]['oa'])
            f1_std = np.std(modus_stats[modus]['f1'])
            print(f"{modus:25s} | OA: {oa_mean:.4f} ± {oa_std:.4f} | F1: {f1_mean:.4f} ± {f1_std:.4f}")
    
    print("\n" + "="*90)
    print("PERFORMANCE SUMMARY BY ARCHITECTURE")
    print("="*90)
    
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
            oa_std = np.std(arch_stats[arch]['oa'])
            f1_std = np.std(arch_stats[arch]['f1'])
            print(f"{arch:20s} | OA: {oa_mean:.4f} ± {oa_std:.4f} | F1: {f1_mean:.4f} ± {f1_std:.4f}")
            
    print("\n" + "="*90)
    print("DETAILED RESULTS")
    print("="*90)
    
    for result in sorted(results, key=lambda x: (x['architecture'], x['modus'])):
        print(f"{result['architecture']:15s} | {result['modus']:25s} | OA: {result['overall_accuracy']:.4f} | F1: {result['mean_f1']:.4f}")

def main():
    """Main function to run the analysis."""
    logger.info("Extracting metrics from today's training results...")
    results = extract_metrics_from_results()
    
    if not results:
        logger.error("No results found. Check if training outputs exist.")
        return
    
    logger.info(f"Found {len(results)} training results")
    
    # Create bar plots
    create_bar_plots(results)

if __name__ == "__main__":
    main()
