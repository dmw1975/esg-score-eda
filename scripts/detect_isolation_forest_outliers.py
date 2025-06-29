#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Isolation Forest Outlier Detection script for ESG Score analysis

This script implements Isolation Forest outlier detection, compares
standard and tuned models, and generates visualizations of the results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Optional, Union

# Import Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.visualization import plot_param_importances, plot_optimization_history
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Warning: Optuna not available. Using simple grid search for tuning.")
    OPTUNA_AVAILABLE = False

# Import project modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location import Location


def detect_standard_isolation_forest_outliers(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42
) -> Tuple[pd.DataFrame, int]:
    """
    Apply standard Isolation Forest algorithm to detect outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    contamination : float, optional
        Proportion of outliers in the data, by default 0.05
    random_state : int, optional
        Random state for reproducibility, by default 42

    Returns
    -------
    Tuple[pd.DataFrame, int]
        DataFrame with outlier flags and total number of outliers
    """
    # Configure Isolation Forest parameters
    isolation_forest = IsolationForest(
        n_estimators=100, 
        contamination=contamination, 
        random_state=random_state
    )

    # Fit the model
    outlier_scores = isolation_forest.fit_predict(df)

    # Create a copy of the original dataframe to store results
    result_df = df.copy()
    
    # Add the outlier flag to the result dataset (-1 indicates an outlier, 1 indicates an inlier)
    result_df.loc[:, 'forest_outlier'] = outlier_scores
    
    # Replace the original dataframe with the result
    df = result_df

    # Count the number of outliers
    total_outliers = np.sum(df['forest_outlier'] == -1)
    
    return df, total_outliers


def tune_isolation_forest_params(
    df: pd.DataFrame,
    params_file: str,
    use_optuna: bool = True,
    n_trials: int = 50,
    timeout: Optional[int] = 600,  # 10 minutes timeout by default
    n_jobs: int = -1  # Use all available cores by default
) -> Dict[str, Union[int, float]]:
    """
    Tune Isolation Forest parameters using Optuna (if available) or grid search.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    params_file : str
        Path to save the JSON file containing the tuned parameters
    use_optuna : bool, default=True
        Whether to use Optuna for hyperparameter optimization
    n_trials : int, default=50
        Number of trials for Optuna optimization
    timeout : int, optional, default=600
        Timeout for optimization in seconds (None means no timeout)
    n_jobs : int, default=-1
        Number of parallel jobs for optimization

    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary of tuned parameters
    """
    # Create a copy of the data and scale it
    df_scaled = StandardScaler().fit_transform(df)
    
    if OPTUNA_AVAILABLE and use_optuna:
        print(f"Tuning Isolation Forest parameters with Optuna (n_trials={n_trials})...")
        
        # Create a study directory for saving Optuna results
        study_dir = os.path.join(os.path.dirname(params_file), 'optuna_studies')
        os.makedirs(study_dir, exist_ok=True)
        db_path = os.path.join(study_dir, 'isolation_forest_study.db')
        
        # Define the objective function for Optuna
        def objective(trial):
            # Define parameter space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'contamination': trial.suggest_float('contamination', 0.005, 0.2),
                'max_samples': trial.suggest_categorical('max_samples', ['auto', 100, 256, 0.1, 0.5, 0.8]),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'n_jobs': n_jobs,  # Use parallel processing for model training
                'random_state': 42
            }
            
            # Create and fit the model
            model = IsolationForest(**params)
            model.fit(df_scaled)
            
            # Get predictions
            predictions = model.predict(df_scaled)
            
            # Evaluation metrics - try multiple approaches
            # 1. Try silhouette score
            try:
                sil_score = silhouette_score(df_scaled, predictions)
                if not np.isfinite(sil_score):
                    sil_score = -1.0
            except:
                sil_score = -1.0  # Default bad score if silhouette fails
                
            # 2. Use mean anomaly score
            anomaly_score = -model.score_samples(df_scaled).mean()
            
            # 3. Count outliers - ensure it's close to expected contamination
            outlier_count = np.sum(predictions == -1)
            outlier_ratio = outlier_count / len(predictions)
            ratio_penalty = abs(outlier_ratio - params['contamination']) * 10
            
            # Combined score (focusing on anomaly_score with small penalty for ratio mismatch)
            score = anomaly_score - ratio_penalty
            
            return score
        
        # Create and run the study
        try:
            study = optuna.create_study(
                study_name='isolation_forest_tuning',
                direction='maximize',
                storage=f'sqlite:///{db_path}',
                load_if_exists=True
            )
            study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1, show_progress_bar=True)
            
            # Get best parameters
            best_params = study.best_params
            best_params['random_state'] = 42  # Ensure reproducibility
            if 'n_jobs' not in best_params:
                best_params['n_jobs'] = n_jobs
                
            print(f"Best parameters (Optuna): {best_params}, Score: {study.best_value:.4f}")
            
            # Generate visualizations if interactive mode
            try:
                # Parameter importance plot
                param_importance_fig = plot_param_importances(study)
                param_importance_path = os.path.join(study_dir, 'param_importance.png')
                param_importance_fig.write_image(param_importance_path, scale=2)
                print(f"Parameter importance plot saved to {param_importance_path}")
                
                # Optimization history plot
                history_fig = plot_optimization_history(study)
                history_path = os.path.join(study_dir, 'optimization_history.png')
                history_fig.write_image(history_path, scale=2)
                print(f"Optimization history plot saved to {history_path}")
            except Exception as viz_error:
                print(f"Warning: Could not generate visualizations: {viz_error}")
                
        except Exception as e:
            print(f"Error during Optuna optimization: {e}")
            print("Falling back to grid search...")
            return _grid_search_tuning(df_scaled, params_file)
            
    else:
        # Fall back to simple grid search if Optuna not available
        print("Using grid search for Isolation Forest parameter tuning...")
        return _grid_search_tuning(df_scaled, params_file)
        
    # Save parameters to file
    os.makedirs(os.path.dirname(params_file), exist_ok=True)
    with open(params_file, 'w') as file:
        json.dump(best_params, file)
    print(f"Best parameters saved to {params_file}")
    
    return best_params


def _grid_search_tuning(df_scaled, params_file):
    """
    Fallback grid search method for tuning Isolation Forest parameters.
    
    Parameters
    ----------
    df_scaled : array-like
        Scaled input data
    params_file : str
        Path to save parameters
        
    Returns
    -------
    dict
        Best parameters
    """
    # Define a simple parameter grid
    param_grid = [
        {
            'n_estimators': n, 
            'contamination': c, 
            'max_samples': s,
            'random_state': 42
        }
        for n in [50, 100, 200, 300]
        for c in [0.01, 0.025, 0.05, 0.075, 0.1]
        for s in ['auto', 256, 0.5, 0.8]
    ]
    
    # Find best parameters
    best_score = float('-inf')
    best_params = None
    
    print(f"Grid searching through {len(param_grid)} parameter combinations...")
    
    # Use joblib for parallel processing if available
    try:
        from joblib import Parallel, delayed
        
        def evaluate_params(params):
            model = IsolationForest(**params)
            model.fit(df_scaled)
            score = -model.score_samples(df_scaled).mean()
            return params, score
        
        results = Parallel(n_jobs=-1)(delayed(evaluate_params)(params) for params in param_grid)
        
        # Find best parameters from results
        for params, score in results:
            if best_params is None or score > best_score:
                best_score = score
                best_params = params
                
    except ImportError:
        # Sequential processing if joblib not available
        for params in param_grid:
            model = IsolationForest(**params)
            model.fit(df_scaled)
            score = -model.score_samples(df_scaled).mean()
            
            if best_params is None or score > best_score:
                best_score = score
                best_params = params
    
    print(f"Best parameters (Grid Search): {best_params}, Score: {best_score:.4f}")
    
    # Save parameters to file
    os.makedirs(os.path.dirname(params_file), exist_ok=True)
    with open(params_file, 'w') as file:
        json.dump(best_params, file)
    print(f"Best parameters saved to {params_file}")
    
    return best_params

def detect_tuned_isolation_forest_outliers(
    df: pd.DataFrame,
    params_file: str,
    tune_params: bool = True,
    use_optuna: bool = True,
    n_trials: int = 50,
    timeout: Optional[int] = 600
) -> Tuple[pd.DataFrame, int]:
    """
    Apply tuned Isolation Forest algorithm to detect outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    params_file : str
        Path to the JSON file containing the tuned parameters
    tune_params : bool, default=True
        Whether to tune parameters if file not found

    Returns
    -------
    Tuple[pd.DataFrame, int]
        DataFrame with tuned outlier flags and total number of outliers
    """
    try:
        # Load the best_params dictionary from the JSON file
        with open(params_file, 'r') as file:
            loaded_params = json.load(file)
        print(f"Loaded parameters from {params_file}")
    except FileNotFoundError:
        if tune_params:
            # Tune parameters and save to file
            loaded_params = tune_isolation_forest_params(
                df=df, 
                params_file=params_file,
                use_optuna=use_optuna,
                n_trials=n_trials,
                timeout=timeout
            )
        else:
            # Use default parameters if file not found and not tuning
            loaded_params = {
                'contamination': 0.01,
                'n_estimators': 100,
                'max_samples': 'auto',
                'random_state': 42
            }
            print(f"Parameters file {params_file} not found. Using default parameters.")

    # Configure Isolation Forest parameters using best_params
    isolation_forest = IsolationForest(
        n_estimators=loaded_params['n_estimators'],
        contamination=loaded_params['contamination'],
        random_state=loaded_params['random_state']
    )

    # Fit the model
    outlier_scores = isolation_forest.fit_predict(df)

    # Create a copy of the original dataframe to store results
    result_df = df.copy()
    
    # Add the outlier flag to the result dataset (-1 indicates an outlier, 1 indicates an inlier)
    result_df.loc[:, 'forest_outlier_tuned'] = outlier_scores

    # Add anomaly scores
    anomaly_scores = isolation_forest.decision_function(df)
    result_df.loc[:, 'anomaly_score'] = anomaly_scores
    
    # Add predicted outlier flag (same as forest_outlier_tuned but easier to interpret)
    result_df.loc[:, 'predicted_outlier'] = outlier_scores
    
    # Replace the original dataframe with the result
    df = result_df

    # Count the number of outliers
    total_outliers_tuned = np.sum(df['forest_outlier_tuned'] == -1)
    
    return df, total_outliers_tuned


def create_outlier_comparison_visualization(
    df: pd.DataFrame,
    save_dir: str
) -> None:
    """
    Create visualizations comparing standard and tuned Isolation Forest outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the outlier detection results
    save_dir : str
        Directory to save the visualizations
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Extract indices of outliers
    outliers = list(df[df['forest_outlier'] == -1].index)
    outliers_tuned = list(df[df['forest_outlier_tuned'] == -1].index)

    # Identify outliers in forest_outlier that are not in outliers_tuned
    exclusive_outliers = set(outliers) - set(outliers_tuned)

    # Get all unique indices for proper axis range
    all_indices = sorted(set(outliers + outliers_tuned))

    # Plot the comparison
    plt.figure(figsize=(8, 15))  # Swapped dimensions for landscape orientation

    # Scatter plot for outliers - note the swapped coordinates
    plt.scatter([1] * len(outliers), outliers, color='red', label='Outliers (forest_outlier)', alpha=0.6, s=50)

    # Scatter plot for outliers_tuned
    plt.scatter([2] * len(outliers_tuned), outliers_tuned, color='blue', label='Outliers (forest_outlier_tuned)', alpha=0.6, s=50)

    # Highlight exclusive outliers
    plt.scatter([1] * len(exclusive_outliers), list(exclusive_outliers), color='green', label='Exclusive (forest_outlier)', alpha=0.8, s=50)

    # Add labels and legend
    plt.xticks([1, 2], ['forest_outlier', 'forest_outlier_tuned'])
    plt.xlabel('Outlier Type')
    plt.ylabel('Index', fontsize=10)
    plt.title('Comparison of Outliers: forest_outlier vs forest_outlier_tuned')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)  # Move legend to bottom
    plt.grid(True)

    # Show all y-axis labels now (previously x-axis)
    plt.yticks(all_indices, all_indices, fontsize=6)

    # Add padding to ensure points near the edges are fully visible
    plt.margins(0.05)

    # Tight layout to ensure everything fits
    plt.tight_layout()

    # Save the figure
    file_name = 'forest_outliers_tuned_landscape.png'
    full_path = os.path.join(save_dir, file_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved at: {os.path.join(save_dir, file_name)}")
    plt.close()

    # Split indices into two approximately equal parts for better readability
    midpoint = len(all_indices) // 2
    first_half_indices = all_indices[:midpoint]
    second_half_indices = all_indices[midpoint:]

    # Create two separate plots for better readability
    for graph_number, indices_subset in enumerate([first_half_indices, second_half_indices], 1):
        plt.figure(figsize=(8, 10))  # Adjusted dimensions for better readability
        
        # Filter outliers for this subset
        subset_outliers = [idx for idx in outliers if idx in indices_subset]
        subset_outliers_tuned = [idx for idx in outliers_tuned if idx in indices_subset]
        subset_exclusive = [idx for idx in exclusive_outliers if idx in indices_subset]
        
        # Scatter plot for outliers
        plt.scatter([1] * len(subset_outliers), subset_outliers, color='red', 
                    label='Outliers (forest_outlier)', alpha=0.6, s=50)
        
        # Scatter plot for outliers_tuned
        plt.scatter([2] * len(subset_outliers_tuned), subset_outliers_tuned, 
                    color='blue', label='Outliers (forest_outlier_tuned)', alpha=0.6, s=50)
        
        # Highlight exclusive outliers
        plt.scatter([1] * len(subset_exclusive), subset_exclusive, 
                    color='green', label='Exclusive (forest_outlier)', alpha=0.8, s=50)
        
        # Add labels and legend
        plt.xticks([1, 2], ['forest_outlier', 'forest_outlier_tuned'])
        plt.xlabel('Outlier Type')
        plt.ylabel('Index', fontsize=10)
        plt.title(f'Comparison of Outliers (Part {graph_number} of 2)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.grid(True)
        
        # Show all y-axis labels for this subset
        plt.yticks(indices_subset, indices_subset, fontsize=7)  # Slightly larger font
        
        # Add padding
        plt.margins(0.05)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the figure
        file_name = f'forest_outliers_tuned_landscape_part{graph_number}.png'
        full_path = os.path.join(save_dir, file_name)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved at: {os.path.join(save_dir, file_name)}")
        plt.close()


def analyze_anomaly_scores(
    df: pd.DataFrame,
    save_dir: str
) -> None:
    """
    Analyze and visualize anomaly scores from Isolation Forest.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data with anomaly scores
    save_dir : str
        Directory to save the visualizations
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Filter the outliers
    outliers = df[df['predicted_outlier'] == -1]
    inliers = df[df['predicted_outlier'] == 1]

    # Create histogram of anomaly scores
    plt.figure(figsize=(12, 6))
    plt.hist(inliers['anomaly_score'], bins=30, color='green', alpha=0.7, label="Inlier Score Distribution")
    plt.hist(outliers['anomaly_score'], bins=30, color='blue', alpha=0.7, label="Outlier Score Distribution")
    plt.axvline(outliers['anomaly_score'].mean(), color='red', linestyle='--', label="Outlier Mean Score")
    plt.title("Distribution of Anomaly Scores", fontsize=16)
    plt.xlabel("Anomaly Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True)

    file_name = 'anomaly_score.png'
    full_path = os.path.join(save_dir, file_name)
    plt.savefig(full_path, dpi=300)
    print(f"Chart saved at: {os.path.join(save_dir, file_name)}")
    plt.close()


def process_isolation_forest_outliers(location: Location, use_optuna: bool = True, n_trials: int = 50, track_times: bool = True) -> Dict[str, float]:
    """
    Process data using Isolation Forest for outlier detection.

    Parameters
    ----------
    location : Location
        Location object for handling file paths
    use_optuna : bool, default=True
        Whether to use Optuna for hyperparameter tuning
    n_trials : int, default=50
        Number of trials for Optuna optimization
    """
    # Dictionary to track execution times of different steps
    step_times = {}
    
    # Try to find the outliers_iqr_z.csv file first
    load_start = time.time() if track_times else 0
    try:
        file_name = 'outliers_iqr_z.csv'
        full_path = location.get_path('data', file_name)
        df = pd.read_csv(full_path, index_col='issuer_name')
        print(f"Using data file: {full_path}")
    except FileNotFoundError:
        # Try the processed directory
        try:
            full_path = location.get_path('data/processed', file_name)
            df = pd.read_csv(full_path, index_col='issuer_name')
            print(f"Using data file: {full_path}")
        except FileNotFoundError:
            # Fall back to imputed_data.csv
            file_name = 'imputed_data.csv'
            full_path = location.get_path('data/processed', file_name)
            df = pd.read_csv(full_path, index_col='issuer_name')
            print(f"Falling back to data file: {full_path}")
    
    if track_times:
        step_times['Data Loading'] = time.time() - load_start

    # Convert to appropriate dtypes
    df = df.convert_dtypes()

    # Preprocessing step - filter columns and prepare data
    preproc_start = time.time() if track_times else 0
    
    # Define parameter scope based on column headings not starting with 'zscore_out_' or 'iqr_out_'
    cols = [col for col in df.columns if not (col.startswith('zscore_out_') or col.startswith('iqr_out_'))]
    
    # Filter to only numeric columns
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude esg_score from analysis as it's the target variable
    if 'esg_score' in numeric_cols:
        numeric_cols.remove('esg_score')
        print("Excluded esg_score from outlier analysis")
    
    print(f"Using {len(numeric_cols)} numeric columns for analysis")
    
    # Create a dataframe with only numeric features
    filtered_df = df[numeric_cols].copy()
    
    if track_times:
        step_times['Data Preprocessing'] = time.time() - preproc_start

    # Apply standard Isolation Forest outlier detection
    std_model_start = time.time() if track_times else 0
    
    filtered_df, total_outliers = detect_standard_isolation_forest_outliers(
        filtered_df,
        contamination=0.05,
        random_state=42
    )
    print(f'Total outliers detected (standard model): {total_outliers}')
    
    if track_times:
        step_times['Standard Isolation Forest'] = time.time() - std_model_start

    # Apply tuned Isolation Forest outlier detection
    tuned_model_start = time.time() if track_times else 0
    
    params_file = location.get_path('visualizations/outliers', 'best_params_isolation_forest.json')
    filtered_df, total_outliers_tuned = detect_tuned_isolation_forest_outliers(
        filtered_df,
        params_file=params_file,
        use_optuna=use_optuna,
        n_trials=n_trials
    )
    print(f'Total outliers detected (tuned model): {total_outliers_tuned}')
    
    if track_times:
        step_times['Tuned Isolation Forest'] = time.time() - tuned_model_start

    # Create visualizations
    viz_start = time.time() if track_times else 0
    
    outliers_dir = location.get_path('visualizations/outliers', 'isolation')
    create_outlier_comparison_visualization(filtered_df, outliers_dir)
    analyze_anomaly_scores(filtered_df, outliers_dir)
    
    if track_times:
        step_times['Visualization Generation'] = time.time() - viz_start

    # Add the forest outlier columns to the original DataFrame and save results
    save_start = time.time() if track_times else 0
    
    df['forest_outlier'] = filtered_df['forest_outlier']
    df['forest_outlier_tuned'] = filtered_df['forest_outlier_tuned']
    df['anomaly_score'] = filtered_df['anomaly_score']
    df['predicted_outlier'] = filtered_df['predicted_outlier']

    # Save results
    output_file = location.get_path('data/processed', 'outlier_all.csv')
    df.to_csv(output_file, index=True)
    print(f"Isolation forest outlier data saved to: {output_file}")

    # Export the list of outliers with details
    outliers = filtered_df[filtered_df['forest_outlier_tuned'] == -1].sort_values('anomaly_score')
    output_file = location.get_path('data/processed', 'isolation_forest_outliers.csv')
    outliers.to_csv(output_file, index=True)
    print(f"Isolation forest outliers (tuned model) saved to: {output_file}")
    
    if track_times:
        step_times['Results Saving'] = time.time() - save_start
        
    # Save step times if tracking enabled
    if track_times:
        time_file = location.get_path('visualizations/performance', 'isolation_forest_times.json')
        os.makedirs(os.path.dirname(time_file), exist_ok=True)
        with open(time_file, 'w') as f:
            json.dump(step_times, f, indent=2)
        print(f"Step execution times saved to: {time_file}")
        
    return step_times


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Isolation Forest Outlier Detection with Optuna Tuning')
    parser.add_argument('--use-optuna', action='store_true', default=True,
                      help='Use Optuna for hyperparameter optimization (default: True)')
    parser.add_argument('--no-optuna', dest='use_optuna', action='store_false',
                      help='Disable Optuna and use grid search instead')
    parser.add_argument('--n-trials', type=int, default=50,
                      help='Number of trials for Optuna optimization (default: 50)')
    parser.add_argument('--timeout', type=int, default=600,
                      help='Timeout for optimization in seconds (default: 600)')
    parser.add_argument('--no-time-tracking', dest='track_times', action='store_false',
                      help='Disable execution time tracking for individual steps')
    
    args = parser.parse_args()
    
    # Initialize location
    base_dir = os.getcwd()    
    location = Location(base_dir)
    
    # Start overall timing
    overall_start = time.time()
    
    # Process data with Isolation Forest
    step_times = process_isolation_forest_outliers(
        location=location,
        use_optuna=args.use_optuna,
        n_trials=args.n_trials,
        track_times=args.track_times
    )
    
    # Calculate overall execution time
    overall_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print("Isolation Forest outlier detection completed successfully.")
    print(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    
    if args.use_optuna and OPTUNA_AVAILABLE:
        print("Used Optuna for hyperparameter tuning.")
    else:
        print("Used grid search for hyperparameter tuning.")
        
    # Print step times summary if time tracking was enabled
    if args.track_times and step_times:
        print("\nStep execution times:")
        total_tracked = sum(step_times.values())
        for step, duration in sorted(step_times.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {step}: {duration:.2f} seconds ({duration/total_tracked*100:.1f}%)")
            
        # Calculate overhead (difference between overall time and sum of step times)
        overhead = overall_time - total_tracked
        print(f"  - Overhead: {overhead:.2f} seconds ({overhead/overall_time*100:.1f}%)")
    
    print("="*80)