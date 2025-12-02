#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def create_oa_barplot():
    """Create comprehensive OA bar plot comparing all modus results."""
    
    # Data extracted from training results
    labels = ['Raw', 'Zero Mean', 'Quantiles', 'Spatial Shuffle 0Mean', 'Spatial Shuffle', 'Std', 'Meanandstd', 'Mean']
    accuracies = [73.12, 52.26, 36.83, 43.88, 69.54, 44.29, 71.83, 71.24]
    
    # Colors: Conv2D_N2 in blue, LinearStatsNet in purple/pink
    colors = ['#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB', '#A23B72', '#A23B72', '#A23B72']
    
    # Network types for legend
    networks = ['Conv2D_N2', 'Conv2D_N2', 'Conv2D_N2', 'Conv2D_N2', 'Conv2D_N2', 'LinearStatsNet', 'LinearStatsNet', 'LinearStatsNet']
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Create bars
    bars = plt.bar(range(len(labels)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.title('Overall Accuracy Comparison Across All Modus Options', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Data Processing Modus', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    
    # Set y-axis limits and ticks
    plt.ylim(0, 90)
    plt.yticks(range(0, 91, 10))
    
    # Set x-axis with consistent rotation
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    
    # Add accuracy values on top of bars
    for i, (bar, acc, net) in enumerate(zip(bars, accuracies, networks)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add separating line between spatial and statistical methods
    plt.axvline(x=4.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add section labels
    plt.text(2, 85, 'Spatial Processing\n(Conv2D_N2)', ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#2E86AB', alpha=0.3),
             fontsize=11, fontweight='bold')
    
    plt.text(6, 85, 'Statistical Processing\n(LinearStatsNet)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#A23B72', alpha=0.3),
             fontsize=11, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Create custom legend at bottom
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', alpha=0.8, label='Conv2D_N2 (Spatial Processing)'),
        Patch(facecolor='#A23B72', alpha=0.8, label='LinearStatsNet (Statistical Features)')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.25), 
               ncol=2, fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to prevent label cutoff and accommodate bottom legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # Add some performance insights as text
    best_spatial = max([(acc, lab) for acc, lab, net in zip(accuracies, labels, networks) if 'Conv2D' in net])
    best_statistical = max([(acc, lab) for acc, lab, net in zip(accuracies, labels, networks) if 'Linear' in net])
    
    plt.figtext(0.02, 0.02, 
                f'Best Spatial: {best_spatial[1]} ({best_spatial[0]:.1f}%) | '
                f'Best Statistical: {best_statistical[1]} ({best_statistical[0]:.1f}%)',
                fontsize=10, style='italic')
    
    # Save and show
    plt.savefig('/home/davideidmann/code/lc_specific_speckle_analysis/oa_comparison_all_modus.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig('/home/davideidmann/code/lc_specific_speckle_analysis/oa_comparison_all_modus.pdf', 
                bbox_inches='tight')
    
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    spatial_accs = [acc for acc, net in zip(accuracies, networks) if 'Conv2D' in net]
    statistical_accs = [acc for acc, net in zip(accuracies, networks) if 'Linear' in net]
    
    print(f"Spatial Processing (Conv2D_N2):")
    print(f"  Mean: {np.mean(spatial_accs):.2f}%")
    print(f"  Std:  {np.std(spatial_accs):.2f}%")
    print(f"  Best: {max(spatial_accs):.2f}% ({labels[accuracies.index(max(spatial_accs))]})")
    print(f"  Worst: {min(spatial_accs):.2f}% ({labels[accuracies.index(min(spatial_accs))]})")
    
    print(f"\nStatistical Processing (LinearStatsNet):")
    print(f"  Mean: {np.mean(statistical_accs):.2f}%")
    print(f"  Std:  {np.std(statistical_accs):.2f}%")
    print(f"  Best: {max(statistical_accs):.2f}% ({labels[accuracies.index(max(statistical_accs))]})")
    print(f"  Worst: {min(statistical_accs):.2f}% ({labels[accuracies.index(min(statistical_accs))]})")
    
    print(f"\nOverall Best: {max(accuracies):.2f}% ({labels[accuracies.index(max(accuracies))]})")
    print(f"Overall Range: {max(accuracies) - min(accuracies):.2f}% spread")

if __name__ == "__main__":
    create_oa_barplot()
