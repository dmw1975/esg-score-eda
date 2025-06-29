#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate ML Output Files

This script generates all necessary files for the ml_output folder structure,
matching the example_data format. It ensures all files are generated exactly once
and placed in the correct subdirectories.
"""

import os
import sys
import shutil
import pandas as pd

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location import Location


def copy_files_to_ml_output(location):
    """
    Copy and organize files from current data directories to ml_output structure.
    
    Parameters
    ----------
    location : Location
        Location object for handling file paths
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    # Create ml_output directories
    ml_output_base = location.get_path('data', 'ml_output')
    ml_output_raw = os.path.join(ml_output_base, 'raw')
    ml_output_pkl = os.path.join(ml_output_base, 'pkl')
    
    # Create directories if they don't exist
    os.makedirs(ml_output_raw, exist_ok=True)
    os.makedirs(ml_output_pkl, exist_ok=True)
    
    print(f"ML Output directories created at: {ml_output_base}")
    
    # Define file mappings (source -> destination)
    file_mappings = {
        # Raw folder files
        location.get_path('data', 'combined_df_for_linear_models.csv'): 
            os.path.join(ml_output_raw, 'combined_df_for_linear_models.csv'),
        
        location.get_path('data', 'combined_df_for_tree_models.csv'): 
            os.path.join(ml_output_raw, 'combined_df_for_tree_models.csv'),
        
        # Score file - try multiple possible locations
        'score': os.path.join(ml_output_raw, 'score.csv'),
        
        # Pkl folder files
        location.get_path('data/processed/pkl', 'base_columns.pkl'): 
            os.path.join(ml_output_pkl, 'base_columns.pkl'),
        
        location.get_path('data/processed/pkl', 'yeo_columns.pkl'): 
            os.path.join(ml_output_pkl, 'yeo_columns.pkl'),
    }
    
    # Special handling for score.csv (could be in multiple locations)
    score_locations = [
        location.get_path('data', 'score.csv'),
        location.get_path('data/processed', 'score.csv'),
    ]
    
    score_copied = False
    for score_path in score_locations:
        if os.path.exists(score_path):
            file_mappings[score_path] = os.path.join(ml_output_raw, 'score.csv')
            score_copied = True
            break
    
    if not score_copied:
        print("Warning: score.csv not found in expected locations")
    
    # Remove the placeholder 'score' entry
    file_mappings.pop('score', None)
    
    # Copy files
    success_count = 0
    error_count = 0
    
    for source, destination in file_mappings.items():
        try:
            if os.path.exists(source):
                shutil.copy2(source, destination)
                print(f"✓ Copied: {os.path.basename(source)} -> {destination}")
                success_count += 1
            else:
                print(f"✗ Source file not found: {source}")
                error_count += 1
        except Exception as e:
            print(f"✗ Error copying {source}: {e}")
            error_count += 1
    
    # Generate summary report
    print("\n" + "="*80)
    print("ML OUTPUT GENERATION SUMMARY")
    print("="*80)
    print(f"Files successfully copied: {success_count}")
    print(f"Errors encountered: {error_count}")
    print(f"ML Output location: {ml_output_base}")
    
    # Verify structure matches example_data
    print("\nVerifying folder structure...")
    expected_files = {
        'raw/combined_df_for_linear_models.csv',
        'raw/combined_df_for_tree_models.csv',
        'raw/score.csv',
        'pkl/base_columns.pkl',
        'pkl/yeo_columns.pkl'
    }
    
    actual_files = set()
    for root, dirs, files in os.walk(ml_output_base):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), ml_output_base)
            actual_files.add(rel_path.replace('\\', '/'))  # Normalize path separators
    
    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
    else:
        print("\n✓ All expected files are present")
    
    if extra_files:
        print(f"\nExtra files found: {extra_files}")
    
    # Create a README for ml_output
    readme_path = os.path.join(ml_output_base, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("ML Output Files for ml_project_refactored\n")
        f.write("="*50 + "\n\n")
        f.write("This folder contains all necessary files for ML model training.\n\n")
        f.write("Structure:\n")
        f.write("- raw/\n")
        f.write("  - combined_df_for_linear_models.csv : Data for Elastic Net models\n")
        f.write("  - combined_df_for_tree_models.csv : Data for tree-based models\n")
        f.write("  - score.csv : ESG scores for evaluation\n")
        f.write("- pkl/\n")
        f.write("  - base_columns.pkl : List of base feature columns\n")
        f.write("  - yeo_columns.pkl : List of Yeo-Johnson transformed columns\n")
        f.write("\nTo use:\n")
        f.write("1. Copy this entire ml_output folder to your ml_project_refactored/data/ directory\n")
        f.write("2. The files will be in the correct structure for the ML pipeline\n")
    
    print(f"\n✓ README created at: {readme_path}")
    
    return error_count == 0


def ensure_all_files_exist(location):
    """
    Ensure all required files exist before copying to ml_output.
    Run missing pipeline steps if necessary.
    
    Parameters
    ----------
    location : Location
        Location object for handling file paths
    """
    print("Checking for required files...")
    
    required_files = {
        'combined_df_for_linear_models.csv': location.get_path('data', 'combined_df_for_linear_models.csv'),
        'combined_df_for_tree_models.csv': location.get_path('data', 'combined_df_for_tree_models.csv'),
        'score.csv': [
            location.get_path('data', 'score.csv'),
            location.get_path('data/processed', 'score.csv')
        ],
        'base_columns.pkl': location.get_path('data/processed/pkl', 'base_columns.pkl'),
        'yeo_columns.pkl': location.get_path('data/processed/pkl', 'yeo_columns.pkl'),
    }
    
    missing_files = []
    
    for file_name, paths in required_files.items():
        if isinstance(paths, list):
            # Check multiple possible locations
            found = any(os.path.exists(p) for p in paths)
            if not found:
                missing_files.append(file_name)
        else:
            # Single location
            if not os.path.exists(paths):
                missing_files.append(file_name)
    
    if missing_files:
        print(f"\nMissing files detected: {missing_files}")
        print("\nThese files should be generated by running the pipeline:")
        print("  python main.py --all")
        print("\nOr specific steps:")
        print("  python main.py --model-specific-data  # For linear/tree model files")
        print("  python main.py --feature-eng          # For score.csv")
        print("  python main.py --yeo-johnson          # For pkl files")
        return False
    else:
        print("\n✓ All required files are present")
        return True


def main():
    """Main function to generate ML output structure."""
    # Initialize location
    base_dir = os.getcwd()
    location = Location(base_dir)
    
    print("ML Output Generation Script")
    print("="*80)
    
    # Check if all required files exist
    if not ensure_all_files_exist(location):
        print("\nPlease run the missing pipeline steps first.")
        return 1
    
    # Copy files to ml_output structure
    if copy_files_to_ml_output(location):
        print("\n✓ ML output generation completed successfully!")
        print("\nNext steps:")
        print("1. Navigate to: data/ml_output/")
        print("2. Copy the entire ml_output folder to your ml_project_refactored/data/ directory")
        print("3. The files are now ready for ML model training")
        return 0
    else:
        print("\n✗ ML output generation encountered errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())