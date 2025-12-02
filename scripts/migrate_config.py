#!/usr/bin/env python3
"""
Configuration Migration Script

Converts existing modus-based configuration files to the new modular
data processing format.
"""

import argparse
import configparser
from pathlib import Path
import sys
import shutil
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lc_speckle_analysis.data_config import _convert_modus_to_modular


def migrate_config_file(input_path: Path, output_path: Path = None, backup: bool = True) -> bool:
    """
    Migrate a single configuration file to new modular format.
    
    Args:
        input_path: Path to existing configuration file
        output_path: Path for new configuration file (default: overwrite input)
        backup: Whether to create backup of original file
        
    Returns:
        True if migration successful, False otherwise
    """
    if not input_path.exists():
        print(f"ERROR: Configuration file not found: {input_path}")
        return False
    
    if output_path is None:
        output_path = input_path
    
    try:
        # Read existing configuration
        config = configparser.ConfigParser()
        config.read(input_path)
        
        # Check if already using new format
        if 'data_processing' in config:
            print(f"INFO: {input_path.name} already uses new modular format")
            return True
        
        # Check if modus or legacy data_with_zero_mean is present
        modus = None
        if 'training_data' in config:
            train_section = config['training_data']
            if 'modus' in train_section:
                modus = train_section.get('modus', 'raw').strip()
            elif 'data_with_zero_mean' in train_section:
                data_with_zero_mean = train_section.getboolean('data_with_zero_mean', False)
                modus = "data_with_zero_mean" if data_with_zero_mean else "raw"
            else:
                modus = "raw"  # Default
        
        if modus is None:
            print(f"WARNING: No modus or data_with_zero_mean found in {input_path.name}, assuming 'raw'")
            modus = "raw"
        
        # Convert modus to modular configuration
        try:
            modular_config = _convert_modus_to_modular(modus)
        except ValueError as e:
            print(f"ERROR: Cannot convert modus '{modus}': {e}")
            return False
        
        # Create backup if requested
        if backup and output_path == input_path:
            backup_path = input_path.with_suffix(f'.bak.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            shutil.copy2(input_path, backup_path)
            print(f"INFO: Created backup: {backup_path.name}")
        
        # Add new data_processing section
        config.add_section('data_processing')
        config.set('data_processing', 'shuffled', str(modular_config.shuffled).lower())
        config.set('data_processing', 'normalized', str(modular_config.normalized).lower())
        config.set('data_processing', 'quantiles', str(modular_config.quantiles).lower())
        if modular_config.aggregation:
            config.set('data_processing', 'aggregation', modular_config.aggregation)
        else:
            config.set('data_processing', 'aggregation', '')
        
        # Remove legacy modus and data_with_zero_mean (but keep as comments)
        if 'training_data' in config:
            train_section = config['training_data']
            legacy_comment = f"\n# Legacy configuration (migrated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n"
            
            if 'modus' in train_section:
                legacy_comment += f"# modus = {train_section.get('modus')}\n"
                train_section.remove_option('modus')
            
            if 'data_with_zero_mean' in train_section:
                legacy_comment += f"# data_with_zero_mean = {train_section.get('data_with_zero_mean')}\n"
                train_section.remove_option('data_with_zero_mean')
        
        # Update network architecture if needed
        if 'neural_network' in config and modular_config.aggregation:
            nn_section = config['neural_network']
            current_arch = nn_section.get('network_architecture_id', '').lower()
            
            if current_arch not in ['linear_stats_net']:
                print(f"INFO: Auto-updating architecture to 'linear_stats_net' for aggregation mode")
                nn_section.set('network_architecture_id', 'linear_stats_net')
        elif 'neural_network' in config and not modular_config.aggregation:
            nn_section = config['neural_network']
            current_arch = nn_section.get('network_architecture_id', '').lower()
            
            if current_arch not in ['test_conv2d', 'test_conv2d_n2']:
                print(f"INFO: Auto-updating architecture to 'test_conv2d_n2' for spatial processing")
                nn_section.set('network_architecture_id', 'test_conv2d_n2')
        
        # Write updated configuration
        with open(output_path, 'w') as f:
            # Add migration header comment
            f.write("# Configuration migrated to modular data processing format\n")
            f.write(f"# Migration date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Original modus: {modus}\n")
            f.write("# New modular parameters:\n")
            f.write(f"#   shuffled = {modular_config.shuffled}\n")
            f.write(f"#   normalized = {modular_config.normalized}\n")
            f.write(f"#   quantiles = {modular_config.quantiles}\n")
            f.write(f"#   aggregation = {modular_config.aggregation or 'None'}\n")
            f.write("\n")
            
            config.write(f)
        
        print(f"SUCCESS: Migrated {input_path.name} -> {output_path.name}")
        print(f"  Original modus: '{modus}'")
        print(f"  New modular config: shuffled={modular_config.shuffled}, "
              f"normalized={modular_config.normalized}, quantiles={modular_config.quantiles}, "
              f"aggregation={modular_config.aggregation or 'None'}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to migrate {input_path.name}: {e}")
        return False


def migrate_directory(directory: Path, pattern: str = "*.conf", backup: bool = True) -> tuple[int, int]:
    """
    Migrate all configuration files in a directory.
    
    Args:
        directory: Directory containing configuration files
        pattern: File pattern to match (default: *.conf)
        backup: Whether to create backups
        
    Returns:
        Tuple of (successful_migrations, total_files)
    """
    if not directory.exists() or not directory.is_dir():
        print(f"ERROR: Directory not found: {directory}")
        return 0, 0
    
    config_files = list(directory.glob(pattern))
    if not config_files:
        print(f"No configuration files found matching '{pattern}' in {directory}")
        return 0, 0
    
    successful = 0
    total = len(config_files)
    
    print(f"Found {total} configuration files to migrate:")
    for config_file in config_files:
        print(f"  {config_file.name}")
    print()
    
    for config_file in config_files:
        if migrate_config_file(config_file, backup=backup):
            successful += 1
        print()  # Add spacing between files
    
    return successful, total


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migrate configuration files to new modular format")
    parser.add_argument('path', type=Path, 
                       help='Configuration file or directory to migrate')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output file (for single file migration only)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup files')
    parser.add_argument('--pattern', default='*.conf',
                       help='File pattern for directory migration (default: *.conf)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be migrated without making changes')
    
    args = parser.parse_args()
    
    if not args.path.exists():
        print(f"ERROR: Path not found: {args.path}")
        sys.exit(1)
    
    backup = not args.no_backup
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print("=" * 50)
    
    if args.path.is_file():
        # Single file migration
        if args.dry_run:
            print(f"Would migrate: {args.path}")
            try:
                config = configparser.ConfigParser()
                config.read(args.path)
                
                if 'data_processing' in config:
                    print("  Already using new format")
                else:
                    modus = "raw"  # Default
                    if 'training_data' in config:
                        train_section = config['training_data']
                        if 'modus' in train_section:
                            modus = train_section.get('modus', 'raw').strip()
                        elif 'data_with_zero_mean' in train_section:
                            data_with_zero_mean = train_section.getboolean('data_with_zero_mean', False)
                            modus = "data_with_zero_mean" if data_with_zero_mean else "raw"
                    
                    modular_config = _convert_modus_to_modular(modus)
                    print(f"  Current modus: '{modus}'")
                    print(f"  Would convert to: shuffled={modular_config.shuffled}, "
                          f"normalized={modular_config.normalized}, quantiles={modular_config.quantiles}, "
                          f"aggregation={modular_config.aggregation or 'None'}")
            except Exception as e:
                print(f"  ERROR: {e}")
        else:
            success = migrate_config_file(args.path, args.output, backup)
            sys.exit(0 if success else 1)
    
    elif args.path.is_dir():
        # Directory migration
        if args.dry_run:
            config_files = list(args.path.glob(args.pattern))
            print(f"Would migrate {len(config_files)} files in {args.path}")
            for config_file in config_files:
                print(f"  {config_file.name}")
        else:
            successful, total = migrate_directory(args.path, args.pattern, backup)
            print(f"Migration complete: {successful}/{total} files migrated successfully")
            sys.exit(0 if successful == total else 1)
    
    else:
        print(f"ERROR: {args.path} is neither a file nor a directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
