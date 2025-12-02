#!/usr/bin/env python3

import json
import glob
import os
import re
from pathlib import Path

def extract_test_accuracy_from_summary(filepath):
    """Extract test accuracy from training summary JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract test accuracy - check the test_results section
        test_acc = None
        if 'test_results' in data and 'test_accuracy' in data['test_results']:
            test_acc = data['test_results']['test_accuracy']
        elif 'test_accuracy' in data:
            test_acc = data['test_accuracy']
        elif 'test_acc' in data:
            test_acc = data['test_acc']
        elif 'results' in data and 'test_accuracy' in data['results']:
            test_acc = data['results']['test_accuracy']
        elif 'results' in data and 'test_acc' in data['results']:
            test_acc = data['results']['test_acc']
        
        return test_acc
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def categorize_run(filepath):
    """Categorize the run based on the filepath."""
    filepath_str = str(filepath)
    
    # Extract modus from the path
    if 'datawithzeromean' in filepath_str:
        modus = 'zero_mean'
    elif 'raw' in filepath_str:
        modus = 'raw'
    elif 'quantiles' in filepath_str:
        modus = 'quantiles'
    elif 'spatial_shuffle_0mean' in filepath_str:
        modus = 'spatial_shuffle_0mean'
    elif 'spatial_shuffle' in filepath_str:
        modus = 'spatial_shuffle'
    elif 'meanandstd' in filepath_str:
        modus = 'meanandstd'
    elif 'mean' in filepath_str and 'meanandstd' not in filepath_str:
        modus = 'mean'
    elif 'std' in filepath_str and 'meanandstd' not in filepath_str:
        modus = 'std'
    else:
        return None, None
    
    # Extract network type
    if 'conv2d_n2' in filepath_str:
        network = 'conv2d_n2'
    elif 'conv2d' in filepath_str:
        network = 'conv2d'
    elif 'linear' in filepath_str:
        network = 'linear'
    else:
        network = 'unknown'
    
    return modus, network

def main():
    # Find all training summary files
    pattern = "/mnt/ssddata/users/david/code/lc_specific_speckle_analysis/data/training_output/*/training_summary*.json"
    files = glob.glob(pattern)
    
    results = {}
    
    for filepath in files:
        modus, network = categorize_run(filepath)
        if modus is None:
            continue
            
        test_acc = extract_test_accuracy_from_summary(filepath)
        if test_acc is None:
            continue
            
        key = f"{modus}_{network}"
        if key not in results or test_acc > results[key]['accuracy']:
            results[key] = {
                'modus': modus,
                'network': network,
                'accuracy': test_acc,
                'filepath': filepath
            }
    
    # Print results in the requested order
    order = [
        ('raw', 'conv2d_n2'),
        ('zero_mean', 'conv2d_n2'), 
        ('quantiles', 'conv2d_n2'),
        ('spatial_shuffle_0mean', 'conv2d_n2'),
        ('spatial_shuffle', 'conv2d_n2'),
        ('std', 'linear'),
        ('meanandstd', 'linear'),
        ('mean', 'linear')
    ]
    
    print("Results for OA bar plot:")
    print("========================")
    
    plot_data = []
    for modus, network in order:
        key = f"{modus}_{network}"
        if key in results:
            acc = results[key]['accuracy'] * 100 if results[key]['accuracy'] < 1 else results[key]['accuracy']
            plot_data.append((modus, network, acc))
            print(f"{modus:20} ({network:10}): {acc:.2f}%")
        else:
            print(f"{modus:20} ({network:10}): NOT FOUND")
    
    # Also create the plotting code
    print("\n" + "="*50)
    print("Python plotting code:")
    print("="*50)
    
    labels = []
    accuracies = []
    colors = []
    
    # Colors: Conv2D_N2 in blue shades, Linear in green shades
    conv2d_color = '#2E86AB'  # Blue
    linear_color = '#A23B72'  # Purple/Pink
    
    for modus, network, acc in plot_data:
        if network == 'conv2d_n2':
            labels.append(modus.replace('_', ' ').title())
            colors.append(conv2d_color)
        else:
            labels.append(modus.replace('_', ' ').title()) 
            colors.append(linear_color)
        accuracies.append(acc)
    
    print(f"labels = {labels}")
    print(f"accuracies = {accuracies}")
    print(f"colors = {colors}")

if __name__ == "__main__":
    main()
