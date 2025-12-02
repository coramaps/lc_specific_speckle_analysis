#!/usr/bin/env python3
"""
Create bar plots comparing Overall Accuracy and mean F1 scores across all modus configurations.
Direct mapping approach using exact directory names.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_metrics_from_results():
    """Extract OA and mean F1 metrics from today's training result directories."""
    
    # Direct mapping of directories to configuration
    training_configs = [
        {
            "directory": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_datawithzeromean_single_20220611_conv2d_5fda7aee",
            "modus": "data_with_zero_mean",
            "architecture": "TestConv2D",
            "label": "Zero Mean + Conv2D"
        },
        {
            "directory": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_datawithzeromean_single_20220611_conv2d_n2_c7877c44",
            "modus": "data_with_zero_mean", 
            "architecture": "TestConv2D_N2",
            "label": "Zero Mean + Conv2D_N2"
        },
        {
            "directory": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_raw_single_20220611_conv2d_6a18e1d7",
            "modus": "raw",
            "architecture": "TestConv2D", 
            "label": "Raw + Conv2D"
        },
        {
            "directory": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_raw_single_20220611_conv2d_n2_0eb5f50f",
            "modus": "raw",
            "architecture": "TestConv2D_N2",
            "label": "Raw + Conv2D_N2"
        },
        {
            "directory": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_quantiles_single_20220611_conv2d_b1f70fec",
            "modus": "quantiles",
            "architecture": "TestConv2D",
            "label": "Quantiles + Conv2D"
        },
        {
            "directory": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_quantiles_single_20220611_conv2d_n2_44f6d712",
            "modus": "quantiles",
            "architecture": "TestConv2D_N2",
            "label": "Quantiles + Conv2D_N2"
        },
        {
            "directory": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_spatialshuffle_single_20220611_conv2d_bbdcd624",
            "modus": "spatial_shuffle",
            "architecture": "TestConv2D",
            "label": "Spatial Shuffle + Conv2D"
        },
        {
            "directory": "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_spatialshuffle_single_20220611_conv2d_n2_3dbe63a0",
            "modus": "spatial_shuffle",
            "architecture": "TestConv2D_N2", 
            "label": "Spatial Shuffle + Conv2D_N2"
        }
    ]
    
    results = []
    
    for config in training_configs:
        run_dir = Path(config["directory"])
        
        logger.info(f"Processing {config['label']} from {run_dir.name}")
        
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
                'modus': config['modus'],
                'architecture': config['architecture'],
                'label': config['label'],
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
    
    # Create labels and values in correct order
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Overall Accuracy plot
    x_pos = np.arange(len(labels))
    bars1 = ax1.bar(x_pos, oa_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1, width=0.6)
    ax1.set_title('Overall Accuracy by Modus and Architecture', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Overall Accuracy', fontsize=14)
    ax1.set_xlabel('Configuration (Modus + Architecture)', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(oa_values) * 1.15)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, oa_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Mean F1 Score plot
    bars2 = ax2.bar(x_pos, f1_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1, width=0.6)
    ax2.set_title('Mean F1 Score by Modus and Architecture', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Mean F1 Score', fontsize=14)
    ax2.set_xlabel('Configuration (Modus + Architecture)', fontsize=14)
    ax2.set_xticks(x_pos)
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
    
    output_file = results_dir / "modus_comparison_metrics_complete.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Bar plots saved to: {output_file}")
    
    plt.show()
    
    # Print comprehensive analysis
    print("\n" + "="*100)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*100)
    
    print("\nINDIVIDUAL RESULTS:")
    print("-" * 80)
    for result in sorted(results, key=lambda x: (x['architecture'], x['modus'])):
        print(f"{result['architecture']:15s} | {result['modus']:25s} | OA: {result['overall_accuracy']:.4f} | F1: {result['mean_f1']:.4f}")
    
    # Group by modus
    print(f"\nPERFORMANCE BY MODUS (averaged across both architectures):")
    print("-" * 80)
    
    modus_stats = {}
    for result in results:
        modus = result['modus']
        if modus not in modus_stats:
            modus_stats[modus] = {'oa': [], 'f1': []}
        modus_stats[modus]['oa'].append(result['overall_accuracy'])
        modus_stats[modus]['f1'].append(result['mean_f1'])
    
    modus_performance = []
    for modus in ['data_with_zero_mean', 'raw', 'quantiles', 'spatial_shuffle']:
        if modus in modus_stats:
            oa_mean = np.mean(modus_stats[modus]['oa'])
            f1_mean = np.mean(modus_stats[modus]['f1'])
            oa_std = np.std(modus_stats[modus]['oa']) if len(modus_stats[modus]['oa']) > 1 else 0
            f1_std = np.std(modus_stats[modus]['f1']) if len(modus_stats[modus]['f1']) > 1 else 0
            modus_performance.append((modus, oa_mean, f1_mean))
            print(f"{modus:25s} | OA: {oa_mean:.4f} ± {oa_std:.4f} | F1: {f1_mean:.4f} ± {f1_std:.4f}")
    
    # Rank by performance
    print(f"\nMODUS RANKING BY OVERALL ACCURACY:")
    print("-" * 50)
    modus_performance_oa = sorted(modus_performance, key=lambda x: x[1], reverse=True)
    for i, (modus, oa, f1) in enumerate(modus_performance_oa, 1):
        modus_display = modus.replace('_', ' ').title().replace('Data With Zero Mean', 'Zero Mean')
        print(f"{i}. {modus_display:20s} ({oa:.4f})")
    
    print(f"\nMODUS RANKING BY MEAN F1 SCORE:")
    print("-" * 50)
    modus_performance_f1 = sorted(modus_performance, key=lambda x: x[2], reverse=True)
    for i, (modus, oa, f1) in enumerate(modus_performance_f1, 1):
        modus_display = modus.replace('_', ' ').title().replace('Data With Zero Mean', 'Zero Mean')
        print(f"{i}. {modus_display:20s} ({f1:.4f})")
    
    # Architecture comparison
    print(f"\nPERFORMANCE BY ARCHITECTURE (averaged across all modus):")
    print("-" * 80)
    
    arch_stats = {}
    for result in results:
        arch = result['architecture']
        if arch not in arch_stats:
            arch_stats[arch] = {'oa': [], 'f1': []}
        arch_stats[arch]['oa'].append(result['overall_accuracy'])
        arch_stats[arch]['f1'].append(result['mean_f1'])
    
    for arch in ['TestConv2D', 'TestConv2D_N2']:
        if arch in arch_stats:
            oa_mean = np.mean(arch_stats[arch]['oa'])
            f1_mean = np.mean(arch_stats[arch]['f1'])
            oa_std = np.std(arch_stats[arch]['oa'])
            f1_std = np.std(arch_stats[arch]['f1'])
            print(f"{arch:20s} | OA: {oa_mean:.4f} ± {oa_std:.4f} | F1: {f1_mean:.4f} ± {f1_std:.4f}")

    # Key insights
    print(f"\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    
    best_oa_modus = modus_performance_oa[0]
    best_f1_modus = modus_performance_f1[0]
    worst_oa_modus = modus_performance_oa[-1]
    worst_f1_modus = modus_performance_f1[-1]
    
    print(f"🏆 Best OA Performance: {best_oa_modus[0].replace('_', ' ').title()} ({best_oa_modus[1]:.4f})")
    print(f"🏆 Best F1 Performance: {best_f1_modus[0].replace('_', ' ').title()} ({best_f1_modus[2]:.4f})")
    print(f"📉 Worst OA Performance: {worst_oa_modus[0].replace('_', ' ').title()} ({worst_oa_modus[1]:.4f})")
    print(f"📉 Worst F1 Performance: {worst_f1_modus[0].replace('_', ' ').title()} ({worst_f1_modus[2]:.4f})")
    
    # Performance gap analysis
    oa_gap = best_oa_modus[1] - worst_oa_modus[1]
    f1_gap = best_f1_modus[2] - worst_f1_modus[2]
    print(f"\n💡 Performance Gap: OA: {oa_gap:.4f} ({oa_gap*100:.1f}%), F1: {f1_gap:.4f} ({f1_gap*100:.1f}%)")
    
    if best_oa_modus[0] in ['raw', 'spatial_shuffle'] and worst_oa_modus[0] == 'quantiles':
        print("📊 Spatial information appears more important than spectral normalization")
        print("   Raw and spatial_shuffle preserve spatial relationships, quantiles destroys them")

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
