#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESG Score EDA Pipeline

This script serves as the main entry point for running the entire ESG Score EDA pipeline
or individual components. It coordinates the execution of all analysis scripts in the
correct order.

Usage:
    python main.py --all                # Run the complete pipeline
    python main.py --missing-values     # Run only missing values analysis
    python main.py --outliers-iqr-z     # Run only IQR and Z-score outlier detection
    python main.py --outliers-forest    # Run only Isolation Forest outlier detection
    python main.py --outlier-plots      # Run only outlier comparison plots
    python main.py --shap-plots        # Generate SHAP visualizations for Isolation Forest
    python main.py --impute-outliers    # Run only outlier imputation
    python main.py --yeo-johnson        # Run only Yeo-Johnson transformation
    python main.py --feature-eng        # Run only feature engineering
    python main.py --model-specific-data # Create optimized data files for linear and tree models
"""

import os
import sys
import argparse
import subprocess
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict

def get_script_path(script_name):
    """Get the absolute path to a script in the scripts directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', script_name)

def run_script(script_name, description, execution_times=None):
    """Run a Python script with proper logging and error handling."""
    script_path = get_script_path(script_name)
    
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print(f"SCRIPT: {script_path}")
    print(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                               check=True, 
                               capture_output=True, 
                               text=True)
        print(result.stdout)
        
        if result.stderr:
            print("WARNINGS:")
            print(result.stderr)
        
        execution_time = time.time() - start_time
        print(f"\nCompleted in {execution_time:.2f} seconds")
        
        # Store execution time if the dictionary is provided
        if execution_times is not None:
            execution_times[description] = execution_time
            
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Script execution failed with code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        # Store failed execution if the dictionary is provided
        if execution_times is not None:
            execution_times[description] = -1  # Negative value indicates failure
            
        return False

def visualize_execution_times(execution_times, save_path=None):
    """Create visualization of pipeline execution times.
    
    Parameters:
    -----------
    execution_times : dict
        Dictionary with step descriptions as keys and execution times as values
    save_path : str, optional
        Path to save the visualization. If None, only displays the plot
    """
    # Skip if no execution times or if all steps failed
    if not execution_times or all(t < 0 for t in execution_times.values()):
        print("No successful execution times to visualize")
        return
    
    # Extract successful executions
    successful_steps = {k: v for k, v in execution_times.items() if v >= 0}
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Sort by execution time (descending)
    sorted_steps = sorted(successful_steps.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_steps]
    times = [item[1] for item in sorted_steps]
    
    # Calculate percentages of total time
    total_time = sum(times)
    percentages = [100 * t / total_time for t in times]
    
    # Create the bar chart
    bars = plt.barh(labels, times, color='skyblue')
    
    # Add execution times and percentages as annotations
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        width = bar.get_width()
        label_text = f"{width:.1f}s ({percentage:.1f}%)"
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 label_text, va='center', fontsize=10)
    
    # Add grid lines for readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add titles and labels
    plt.title('Pipeline Step Execution Times', fontsize=16, fontweight='bold')
    plt.xlabel('Execution Time (seconds)', fontsize=12)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nExecution time visualization saved to: {save_path}")
    
    # Close the plot to avoid display in non-interactive environments
    plt.close()


def save_execution_times(execution_times, save_path):
    """Save execution times to a JSON file.
    
    Parameters:
    -----------
    execution_times : dict
        Dictionary with step descriptions as keys and execution times as values
    save_path : str
        Path to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Add timestamp
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'execution_times': execution_times,
        'total_time': sum(t for t in execution_times.values() if t >= 0)
    }
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nExecution times saved to: {save_path}")


def print_execution_summary(execution_times):
    """Print a summary of execution times.
    
    Parameters:
    -----------
    execution_times : dict
        Dictionary with step descriptions as keys and execution times as values
    """
    # Skip if no execution times
    if not execution_times:
        return
    
    # Calculate total time for successful steps
    successful_times = [t for t in execution_times.values() if t >= 0]
    total_time = sum(successful_times)
    
    # Count failed steps
    failed_steps = [k for k, v in execution_times.items() if v < 0]
    
    # Sort steps by execution time (descending)
    sorted_steps = sorted([(k, v) for k, v in execution_times.items() if v >= 0], 
                          key=lambda x: x[1], reverse=True)
    
    # Print the summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION TIME SUMMARY")
    print("="*80)
    
    # Print the top 3 most time-consuming steps
    if sorted_steps:
        print("\nMost time-consuming steps:")
        for i, (step, time_taken) in enumerate(sorted_steps[:3], 1):
            percentage = 100 * time_taken / total_time
            print(f"{i}. {step}: {time_taken:.2f} seconds ({percentage:.1f}%)")
    
    # Print failed steps if any
    if failed_steps:
        print("\nFailed steps:")
        for step in failed_steps:
            print(f"- {step}")
    
    # Print total execution time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*80)


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="ESG Score EDA Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Add arguments for each pipeline step
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    parser.add_argument('--missing-values', action='store_true', help='Run missing values analysis')
    parser.add_argument('--outliers-iqr-z', action='store_true', help='Run IQR and Z-score outlier detection')
    parser.add_argument('--outliers-forest', action='store_true', help='Run Isolation Forest outlier detection')
    parser.add_argument('--outlier-plots', action='store_true', help='Run outlier comparison plots')
    parser.add_argument('--shap-plots', action='store_true', help='Generate SHAP visualizations for Isolation Forest')
    parser.add_argument('--impute-outliers', action='store_true', help='Run outlier imputation')
    parser.add_argument('--yeo-johnson', action='store_true', help='Apply Yeo-Johnson transformation')
    parser.add_argument('--feature-eng', action='store_true', help='Run feature engineering')
    parser.add_argument('--model-specific-data', action='store_true', help='Create model-specific data files (linear vs tree models)')
    parser.add_argument('--ml-output', action='store_true', help='Generate ML output folder structure for ml_project_refactored')
    
    # Add time tracking arguments
    parser.add_argument('--no-time-tracking', action='store_true', help='Disable execution time tracking')
    parser.add_argument('--time-viz-path', type=str, default='visualizations/performance/execution_times.png',
                      help='Path to save the execution time visualization')
    parser.add_argument('--time-json-path', type=str, default='visualizations/performance/execution_times.json',
                      help='Path to save the execution time data as JSON')
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Define the pipeline steps
    pipeline_steps = [
        ('analyze_missing_values.py', 'Missing Values Analysis', args.missing_values),
        ('detect_outliers.py', 'IQR and Z-Score Outlier Detection', args.outliers_iqr_z),
        ('detect_isolation_forest_outliers.py', 'Isolation Forest Outlier Detection', args.outliers_forest),
        ('generate_isolation_forest_plots.py', 'Isolation Forest Visualizations', args.outliers_forest),
        ('generate_outlier_comparison_plots.py', 'Outlier Comparison Plots', args.outlier_plots),
        # SHAP visualizations are now generated during the Isolation Forest detection step
        # and saved in visualizations/outliers/shap/tuned_model/
        ('impute_outliers.py', 'Outlier Imputation', args.impute_outliers),
        ('apply_yeo_johnson.py', 'Yeo-Johnson Transformation', args.yeo_johnson),
        ('feature_engineering.py', 'Feature Engineering', args.feature_eng),
        ('create_ml_model_data.py', 'Create ML Model Data', args.feature_eng),
        ('generate_score.py', 'Generate Score CSV', args.feature_eng),
        ('create_model_specific_data.py', 'Create Model-Specific Data Files', args.model_specific_data),
        ('generate_ml_output.py', 'Generate ML Output Structure', args.ml_output),
        ('generate_metadata.py', 'Generate ML Metadata JSON Files', args.ml_output)
    ]
    
    # Run either all steps or the selected ones
    steps_to_run = [(s, d) for s, d, f in pipeline_steps if args.all or f]
    
    # Initialize execution times dictionary if time tracking is enabled
    execution_times = OrderedDict() if not args.no_time_tracking else None
    
    # Record start time of the entire pipeline
    pipeline_start_time = time.time()
    print(f"Starting ESG Score EDA Pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run each step in the pipeline
    for script_name, description in steps_to_run:
        if not run_script(script_name, description, execution_times):
            print(f"\nERROR: Pipeline failed at step: {description}")
            print("Exiting pipeline execution.")
            
            # Save execution times even if pipeline fails
            if execution_times is not None:
                print_execution_summary(execution_times)
                save_execution_times(execution_times, args.time_json_path)
                visualize_execution_times(execution_times, args.time_viz_path)
            
            return 1
    
    # Calculate total pipeline execution time
    pipeline_execution_time = time.time() - pipeline_start_time
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TOTAL EXECUTION TIME: {pipeline_execution_time:.2f} seconds ({pipeline_execution_time/60:.2f} minutes)")
    print("="*80 + "\n")
    
    # If time tracking is enabled, print summary and save results
    if execution_times is not None:
        # Add the total pipeline time (includes overhead)
        execution_times['Pipeline Overhead'] = pipeline_execution_time - sum(t for t in execution_times.values() if t >= 0)
        
        # Print summary, save data, and create visualization
        print_execution_summary(execution_times)
        save_execution_times(execution_times, args.time_json_path)
        visualize_execution_times(execution_times, args.time_viz_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())