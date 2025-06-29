#!/usr/bin/env python3
"""
Outlier Detection Module for ESG Score Data Analysis

This module provides functions for detecting and processing outliers
using various methods including IQR, Z-score, and Isolation Forest.
"""

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logger = logging.getLogger(__name__)

def identify_numerical_columns(df):
    """
    Identify numerical columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
        
    Returns:
    --------
    pandas.Index
        Column names of numerical variables
    """
    return df.select_dtypes(include=['float64', 'int64', 'Float64', 'Int64']).columns


def plot_boxplots(df, numerical_columns, features_per_plot=9, save_path=None, dpi=300):
    """
    Create boxplots of numerical variables to visualize distributions and potential outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the data
    numerical_columns : list or pandas.Index
        List of numerical column names to plot
    features_per_plot : int, default=9
        Number of features to display in each plot grid
    save_path : str, default=None
        Directory path to save the plots. If None, plots are only displayed
    dpi : int, default=300
        DPI for saved plots
    
    Returns:
    --------
    list
        List of created figure objects
    """
    # Number of plots
    num_plots = math.ceil(len(numerical_columns) / features_per_plot)
    figures = []
    
    # Set font size - make it larger for better readability
    plt.rcParams.update({'font.size': 12})
    
    # Loop through the groups of features
    for plot_num in range(num_plots):
        fig = plt.figure(figsize=(15, 15))
        figures.append(fig)
        
        # Determine current batch of features
        start_idx = plot_num * features_per_plot
        end_idx = min((plot_num + 1) * features_per_plot, len(numerical_columns))
        current_features = numerical_columns[start_idx:end_idx]
        
        # Create subplots
        for index, feature in enumerate(current_features, start=1):
            plt.subplot(3, 3, index)
            sns.boxplot(y=feature, data=df, notch=True)
            plt.title(feature, fontsize=14, fontweight='bold')  # Larger, bold title
            plt.xlabel('', fontsize=10)
            plt.ylabel('', fontsize=10)
            plt.yticks(fontsize=10)  # Larger axis tick labels
            
            # Adjust scientific notation font size
            ax = plt.gca()
            ax.yaxis.get_offset_text().set_fontsize(10)
                
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            file_name = f'boxplot_group_{plot_num + 1}.png'
            full_path = os.path.join(save_path, file_name)
            plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Boxplot saved to {full_path}")
    
    return figures


#--------------- IQR-based Outlier Detection ---------------#

def detect_iqr_outliers(df, columns=None):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list, default=None
        List of column names to check for outliers. If None, uses all numerical columns
    
    Returns:
    --------
    tuple
        (outlier_flags, outlier_counts, outlier_details)
        - outlier_flags: DataFrame with boolean flags indicating outliers
        - outlier_counts: Series with count of outliers in each column
        - outlier_details: Dict with lower and upper tail information
    """
    if columns is None:
        columns = identify_numerical_columns(df)
    
    # Calculate Q1, Q3, and IQR
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_flags = ((df[columns] < lower_bound) | (df[columns] > upper_bound))
    
    # Count outliers
    outlier_counts = outlier_flags.sum()
    
    # Get details on the tails
    lower_tails = {}
    upper_tails = {}
    for col in columns:
        lower_tails[col] = df[col][df[col] < lower_bound[col]]
        upper_tails[col] = df[col][df[col] > upper_bound[col]]
    
    outlier_details = {
        'lower_tails': lower_tails,
        'upper_tails': upper_tails,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'lower_count': sum(len(lower_tails[col]) for col in columns),
        'upper_count': sum(len(upper_tails[col]) for col in columns),
        'total_count': sum(len(lower_tails[col]) + len(upper_tails[col]) for col in columns)
    }
    
    return outlier_flags, outlier_counts, outlier_details


def add_iqr_outlier_flags(df, columns=None, prefix='iqr_out_'):
    """
    Add columns to the DataFrame indicating IQR outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list, default=None
        List of column names to check for outliers. If None, uses all numerical columns
    prefix : str, default='iqr_out_'
        Prefix for the new columns that will be added
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added outlier flag columns
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = identify_numerical_columns(df_copy)
    
    for col in columns:
        # Calculate Q1, Q3, and IQR
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Identify outliers
        is_outlier = (df_copy[col] < (Q1 - 1.5 * IQR)) | (df_copy[col] > (Q3 + 1.5 * IQR))
        
        # Flag outliers (1 for outliers, 0 for non-outliers)
        df_copy[f'{prefix}{col}'] = is_outlier.astype(int)
    
    return df_copy


#--------------- Z-Score Outlier Detection ---------------#

def detect_zscore_outliers(df, columns=None, threshold=3):
    """
    Detect outliers using the Z-score method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list, default=None
        List of column names to check for outliers. If None, uses all numerical columns
    threshold : float, default=3
        Z-score threshold beyond which a value is considered an outlier
    
    Returns:
    --------
    tuple
        (outlier_flags, outlier_counts, outlier_details)
        - outlier_flags: DataFrame with boolean flags indicating outliers
        - outlier_counts: Series with count of outliers in each column
        - outlier_details: Dict with lower and upper tail information
    """
    if columns is None:
        columns = identify_numerical_columns(df)
    
    # Calculate Z-scores for each column
    z_scores = df[columns].apply(zscore, nan_policy='omit')
    
    # Identify outliers as values with Z-scores beyond the threshold
    outlier_flags = (z_scores.abs() > threshold)
    
    # Count outliers
    outlier_counts = outlier_flags.sum()
    
    # Get details on the tails
    lower_tails = {}
    upper_tails = {}
    for col in columns:
        lower_tails[col] = df[col][z_scores[col] < -threshold]
        upper_tails[col] = df[col][z_scores[col] > threshold]
    
    outlier_details = {
        'lower_tails': lower_tails,
        'upper_tails': upper_tails,
        'lower_count': sum(len(lower_tails[col]) for col in columns),
        'upper_count': sum(len(upper_tails[col]) for col in columns),
        'total_count': sum(len(lower_tails[col]) + len(upper_tails[col]) for col in columns)
    }
    
    return outlier_flags, outlier_counts, outlier_details


def add_zscore_outlier_flags(df, columns=None, threshold=3, prefix='zscore_out_'):
    """
    Add columns to the DataFrame indicating Z-score outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list, default=None
        List of column names to check for outliers. If None, uses all numerical columns
    threshold : float, default=3
        Z-score threshold beyond which a value is considered an outlier
    prefix : str, default='zscore_out_'
        Prefix for the new columns that will be added
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added outlier flag columns
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = identify_numerical_columns(df_copy)
    
    # Calculate Z-scores
    z_scores = df_copy[columns].apply(zscore, nan_policy='omit')
    
    # Flag outliers
    outlier_flags = (z_scores.abs() > threshold)
    
    # Add flag columns
    for col in columns:
        df_copy[f'{prefix}{col}'] = outlier_flags[col].astype(int)
    
    return df_copy


#--------------- Isolation Forest Outlier Detection ---------------#

def detect_isolation_forest_outliers(df, n_estimators=100, contamination=0.05, 
                                     random_state=42, features=None, prefix='forest_outlier',
                                     add_score=True):
    """
    Detect outliers using the Isolation Forest algorithm.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    n_estimators : int, default=100
        Number of base estimators in the ensemble
    contamination : float, default=0.05
        The proportion of outliers in the data set (between 0 and 0.5)
    random_state : int, default=42
        Random seed for reproducibility
    features : list, default=None
        List of column names to use for outlier detection. If None, all numeric columns are used
    prefix : str, default='forest_outlier'
        Column name for the outlier flags
    add_score : bool, default=True
        Whether to add anomaly scores to the output DataFrame
        
    Returns:
    --------
    tuple
        (df_copy, iso_forest)
        - df_copy: DataFrame with added outlier columns
        - iso_forest: Fitted Isolation Forest model
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("scikit-learn is required for Isolation Forest detection")
        raise ImportError("scikit-learn is required for Isolation Forest detection")
    
    # Select features
    if features is None:
        features = identify_numerical_columns(df)
    
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Extract features for the model
    X = df_copy[features].fillna(df_copy[features].mean())  # Handle missing values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit the Isolation Forest model
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    
    # Fit the model and get predictions
    outlier_predictions = iso_forest.fit_predict(X_scaled)
    
    # In sklearn, -1 is outlier and 1 is inlier, so we need to convert to our convention
    # where 1 is outlier and 0 is inlier
    df_copy[prefix] = (outlier_predictions == -1).astype(int)
    
    # Get anomaly scores if requested
    if add_score:
        # decision_function returns low values for outliers, high for inliers
        # we invert it so high values are outliers
        df_copy['anomaly_score'] = -iso_forest.decision_function(X_scaled)
    
    return df_copy, iso_forest  # Return both the dataframe and the fitted model


def detect_isolation_forest_tuned_outliers(df, features=None, param_grid=None, prefix='forest_out_tuned_',
                                           add_score=True, save_path=None, use_optuna=True, n_trials=50, 
                                           timeout=600):
    """
    Detect outliers using a tuned Isolation Forest algorithm.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    features : list, default=None
        List of column names to use for outlier detection. If None, all numeric columns are used
    param_grid : dict, default=None
        Dictionary with parameter names as keys and lists of parameter values to tune.
        If None, a default grid will be used
    prefix : str, default='forest_out_tuned_'
        Column name for the tuned outlier flags
    add_score : bool, default=True
        Whether to add anomaly scores to the output DataFrame
    save_path : str, default=None
        Path to save the best parameters JSON file. If None, parameters are not saved
        
    Returns:
    --------
    tuple
        (df_copy, best_params, iso_forest)
        - df_copy: DataFrame with added outlier columns
        - best_params: Dictionary with best parameters found
        - iso_forest: Fitted Isolation Forest model
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        import json
        from sklearn.model_selection import ParameterGrid
        from sklearn.metrics import silhouette_score
    except ImportError:
        logger.error("scikit-learn is required for tuned Isolation Forest detection")
        raise ImportError("scikit-learn is required for tuned Isolation Forest detection")
    
    # Select features
    if features is None:
        features = identify_numerical_columns(df)
    
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Extract features for the model
    X = df_copy[features].fillna(df_copy[features].mean())  # Handle missing values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'contamination': [0.01, 0.05, 0.1],
            'random_state': [42]
        }
    
    # Check if Optuna is available
    try:
        import optuna
        from optuna.visualization import plot_param_importances, plot_optimization_history
        optuna_available = True
    except ImportError:
        logger.info("Optuna not installed, falling back to grid search.")
        optuna_available = False
        
    # Use Optuna for hyperparameter tuning if available and enabled
    if optuna_available and use_optuna:
        logger.info(f"Tuning Isolation Forest with Optuna (n_trials={n_trials})...")
        
        # Create a study directory for saving Optuna results if save_path is provided
        study_dir = None
        if save_path:
            study_dir = os.path.join(save_path, 'optuna_studies')
            os.makedirs(study_dir, exist_ok=True)
            storage_path = os.path.join(study_dir, 'isolation_forest_study.db')
            storage = f"sqlite:///{storage_path}"
        else:
            storage = None
            
        # Define objective function for Optuna
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'contamination': trial.suggest_float('contamination', 0.005, 0.2),
                'max_samples': trial.suggest_categorical('max_samples', ['auto', 100, 256, 0.1, 0.5, 0.8]),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }
            
            # Create and train the model
            model = IsolationForest(**params)
            model.fit(X_scaled)
            
            # Get predictions
            predictions = model.predict(X_scaled)
            
            # Try silhouette score
            try:
                score = silhouette_score(X_scaled, predictions)
                return score
            except ValueError:
                # Use alternative scoring if silhouette fails
                return -model.score_samples(X_scaled).mean()
                
        # Create and run study
        try:
            study = optuna.create_study(
                study_name='isolation_forest_tuning',
                direction='maximize',
                storage=storage,
                load_if_exists=True
            )
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            # Get best parameters
            best_params = study.best_params
            best_params['random_state'] = 42  # Ensure reproducibility
            best_score = study.best_value
            
            logger.info(f"Best Optuna parameters: {best_params}, Score: {best_score:.4f}")
            
            # Save study visualizations if save_path is provided
            if save_path and study_dir:
                try:
                    # Save parameter importance plot
                    fig = plot_param_importances(study)
                    fig_path = os.path.join(study_dir, 'param_importance.png')
                    fig.write_image(fig_path)
                    logger.info(f"Parameter importance plot saved to {fig_path}")
                    
                    # Save optimization history plot
                    fig = plot_optimization_history(study)
                    fig_path = os.path.join(study_dir, 'optimization_history.png')
                    fig.write_image(fig_path)
                    logger.info(f"Optimization history plot saved to {fig_path}")
                except Exception as e:
                    logger.warning(f"Error creating Optuna visualizations: {e}")
                    
        except Exception as e:
            logger.error(f"Error during Optuna optimization: {e}")
            logger.info("Falling back to grid search...")
            # Fall back to grid search
            best_params, best_score = _grid_search_isolation_forest(param_grid, X_scaled)
    else:
        # Use grid search
        logger.info("Using grid search for Isolation Forest tuning...")
        best_params, best_score = _grid_search_isolation_forest(param_grid, X_scaled)
        
    logger.info(f"Best parameters: {best_params}, Score: {best_score:.4f}")
    
    logger.info(f"Best parameters: {best_params}, Score: {best_score:.4f}")
    
    # Save best parameters if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, 'best_params_isolation_forest.json')
        with open(file_path, 'w') as file:
            json.dump(best_params, file)
        logger.info(f"Best parameters saved to {file_path}")
    
    # Use the best parameters to fit the final model
    iso_forest = IsolationForest(**best_params)
    
    # Fit the model and get predictions
    outlier_predictions = iso_forest.fit_predict(X_scaled)
    
    # Add outlier flags to DataFrame
    df_copy[prefix] = (outlier_predictions == -1).astype(int)
    
    # Get anomaly scores if requested
    if add_score:
        df_copy['anomaly_score_tuned'] = -iso_forest.decision_function(X_scaled)
    
    return df_copy, best_params, iso_forest


def _grid_search_isolation_forest(param_grid, X_scaled):
    """
    Helper function to perform grid search for Isolation Forest parameters.
    
    Parameters:
    -----------
    param_grid : dict
        Parameter grid to search
    X_scaled : array-like
        Scaled feature matrix
        
    Returns:
    --------
    tuple
        (best_params, best_score)
    """
    best_params = None
    best_score = -float('inf')
    
    # Iterate through parameter combinations
    for params in ParameterGrid(param_grid):
        # Configure Isolation Forest with current parameters
        isolation_forest = IsolationForest(**params)
        
        # Fit the model
        isolation_forest.fit(X_scaled)
        
        # Calculate anomaly scores
        scores = isolation_forest.decision_function(X_scaled)
        
        # Use silhouette score as a proxy for cluster separation
        try:
            current_score = silhouette_score(X_scaled, isolation_forest.predict(X_scaled))
        except ValueError:
            current_score = -float('inf')  # Assign worst score if silhouette fails
        
        # Update best parameters if the current score is better
        if current_score > best_score:
            best_score = current_score
            best_params = params
            
    return best_params, best_score


def tune_isolation_forest(df, param_grid=None, features=None, cv=3, scoring='neg_mean_squared_error', use_optuna=True, n_trials=50):
    """
    Tune hyperparameters for Isolation Forest.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    param_grid : dict, default=None
        Dictionary with parameter names as keys and lists of parameter values.
        If None, a default grid will be used
    features : list, default=None
        List of column names to use for outlier detection. If None, all numeric columns are used
    cv : int, default=3
        Number of cross-validation folds
    scoring : str, default='neg_mean_squared_error'
        Scoring metric for parameter selection
        
    Returns:
    --------
    dict
        Best parameters found during tuning
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV
    except ImportError:
        logger.error("scikit-learn is required for Isolation Forest tuning")
        raise ImportError("scikit-learn is required for Isolation Forest tuning")
    
    # Select features
    if features is None:
        features = identify_numerical_columns(df)
    
    # Extract features for the model
    X = df[features].fillna(df[features].mean())  # Handle missing values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Default parameter grid if none is provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'contamination': [0.01, 0.05, 0.1],
            'max_samples': ['auto', 100, 0.1, 0.5],
            'random_state': [42]
        }
    
    # Initialize and fit the GridSearchCV
    iso_forest = IsolationForest()
    grid_search = GridSearchCV(
        iso_forest, 
        param_grid=param_grid, 
        cv=cv, 
        scoring=scoring,
        n_jobs=-1  # Use all available processors
    )
    
    # Fit the grid search
    grid_search.fit(X_scaled)
    
    return grid_search.best_params_


#--------------- Visualization Functions ---------------#

def plot_outlier_comparison(iqr_outliers, zscore_outliers, forest_outliers=None, save_path=None, dpi=300, include_isolation_forest=False):
    """
    Plot a comparison of outliers detected by different methods.
    
    Parameters:
    -----------
    iqr_outliers : pandas.Series
        Series containing the count of outliers detected by IQR method
    zscore_outliers : pandas.Series
        Series containing the count of outliers detected by Z-score method
    forest_outliers : pandas.Series, default=None
        Series containing the count of outliers detected by Isolation Forest
    save_path : str, default=None
        Path to save the plot. If None, plot is only displayed
    dpi : int, default=300
        DPI for saved plot
    include_isolation_forest : bool, default=False
        Whether to include Isolation Forest results in the comparison
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Create comparison dataframe
    comparison = {'IQR Outliers': iqr_outliers, 'Z-Score Outliers': zscore_outliers}
    if forest_outliers is not None and include_isolation_forest:
        comparison['Isolation Forest Outliers'] = forest_outliers
    
    comparison_df = pd.DataFrame(comparison)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_title('Comparison of Outliers Detected by Different Methods')
    ax.set_ylabel('Number of Outliers')
    ax.set_xlabel('Variables')
    ax.set_xticks(range(len(comparison_df.index)))
    ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Comparison chart saved to {save_path}")
    
    return fig


def plot_isolation_forest_outliers(df, original_column=None, tuned_column=None, save_path=None, dpi=300):
    """
    Visualize outliers detected by Isolation Forest original and tuned models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with outlier columns
    original_column : str, default=None
        Column name for original Isolation Forest outliers (e.g., 'forest_outlier')
    tuned_column : str, default=None
        Column name for tuned Isolation Forest outliers (e.g., 'forest_out_tuned_')
    save_path : str, default=None
        Path to save the visualizations. If None, plots are only displayed.
    dpi : int, default=300
        DPI for saved visualizations
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Skip if both column names are None
    if original_column is None and tuned_column is None:
        logger.warning("No outlier columns specified for visualization")
        return None
    
    # Extract outlier indices
    outliers_indices = []
    labels = []
    
    if original_column is not None and original_column in df.columns:
        original_outliers = list(df[df[original_column] == 1].index)
        outliers_indices.append(original_outliers)
        labels.append(original_column)
    
    if tuned_column is not None and tuned_column in df.columns:
        tuned_outliers = list(df[df[tuned_column] == 1].index)
        outliers_indices.append(tuned_outliers)
        labels.append(tuned_column)
    
    # Find exclusive outliers (in original but not in tuned)
    if len(outliers_indices) == 2:
        exclusive_outliers = list(set(outliers_indices[0]) - set(outliers_indices[1]))
        outliers_indices.append(exclusive_outliers)
        labels.append(f"Exclusive ({original_column})")
    
    # Get all unique indices for proper axis range
    all_indices = sorted(set(sum(outliers_indices, [])))
    
    # Create the visualization based on number of outlier types
    if len(outliers_indices) >= 2:
        # For landscape orientation (better for comparing methods)
        fig = plt.figure(figsize=(6, max(10, min(30, 4 + len(all_indices) * 0.4))))
        ax = fig.add_subplot(111)
        
        # Create scatter points for each outlier type
        colors = ['red', 'blue', 'green']
        for i, (indices, label) in enumerate(zip(outliers_indices, labels)):
            ax.scatter([i+1] * len(indices), indices, color=colors[i], label=label, alpha=0.6, s=50)
        
        # Set plot properties
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Outlier Type')
        ax.set_ylabel('Index')
        ax.set_yticks(all_indices)
        ax.set_yticklabels(all_indices, fontsize=6)
        ax.grid(True)
        ax.set_title('Comparison of Outliers: Original vs Tuned Isolation Forest')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        plt.tight_layout()
    else:
        # Basic visualization for a single outlier type
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(all_indices, [1] * len(all_indices), color='red', label=labels[0])
        ax.set_title(f'Outliers Detected by {labels[0]}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Outlier Status')
        ax.set_xticks(all_indices)
        ax.set_xticklabels(all_indices, rotation=90, fontsize=8)
        plt.tight_layout()
    
    # Save visualization if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Isolation Forest comparison saved to {save_path}")
    
    return fig


def plot_shap_values(model, X, feature_names=None, max_display=30, save_path=None, dpi=300):
    """
    Create SHAP value visualizations for an Isolation Forest model to explain outlier detection.
    
    Parameters:
    -----------
    model : IsolationForest
        Fitted Isolation Forest model
    X : array-like
        Feature matrix used for the visualization
    feature_names : list, default=None
        Names of the features in X. If None, uses X.columns if X is a DataFrame, else uses feature indices
    max_display : int, default=30
        Maximum number of features to display in the summary plot
    save_path : str, default=None
        Path to save the visualizations. If None, plots are only displayed.
    dpi : int, default=300
        DPI for saved visualizations
        
    Returns:
    --------
    dict
        Dictionary containing the created figure objects
    """
    try:
        import shap
        # Fix for NumPy 2.0 compatibility - monkey patch the obj2sctype function if needed
        if hasattr(np, 'version') and int(np.version.version.split('.')[0]) >= 2:
            if not hasattr(np, 'obj2sctype'):
                # Create a compatible version of obj2sctype using np.dtype
                def obj2sctype_compat(obj):
                    try:
                        return np.dtype(obj).type
                    except (TypeError, ValueError):
                        return None
                np.obj2sctype = obj2sctype_compat
                logger.info("Added NumPy 2.0 compatibility for SHAP visualizations")
    except ImportError:
        logger.error("shap is required for SHAP value visualizations")
        raise ImportError("shap is required for SHAP value visualizations")
    
    figures = {}
    
    # If X is a DataFrame, get feature names and convert to numpy array
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X_values = X.values
    else:
        X_values = X
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X_values.shape[1])]
    
    try:
        # Initialize the SHAP explainer for the Isolation Forest model
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_values)
        
        # 1. Summary plot (dot plot)
        plt.figure(figsize=(10, max(8, min(30, 4 + len(feature_names) * 0.4))))
        shap.summary_plot(shap_values, X_values, feature_names=feature_names, max_display=max_display, show=False)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            summary_path = os.path.join(save_path, 'shap_summary_dot.png')
            plt.savefig(summary_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"SHAP summary dot plot saved to {summary_path}")
        
        figures['summary_dot'] = plt.gcf()
        plt.close()
        
        # 2. Bar plot of feature importance
        plt.figure(figsize=(10, max(8, min(20, 4 + len(feature_names) * 0.2))))
        shap.summary_plot(shap_values, X_values, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            bar_path = os.path.join(save_path, 'shap_summary_bar.png')
            plt.savefig(bar_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"SHAP summary bar plot saved to {bar_path}")
        
        figures['summary_bar'] = plt.gcf()
        plt.close()
        
        # 3. Decision plot (if dataset is not too large)
        if X_values.shape[0] <= 1000:  # Limit to avoid performance issues
            plt.figure(figsize=(10, 12))
            shap.decision_plot(explainer.expected_value, shap_values, feature_names=feature_names, show=False)
            plt.tight_layout()
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                decision_path = os.path.join(save_path, 'shap_decision_plot.png')
                plt.savefig(decision_path, dpi=dpi, bbox_inches='tight')
                logger.info(f"SHAP decision plot saved to {decision_path}")
            
            figures['decision_plot'] = plt.gcf()
            plt.close()
    except Exception as e:
        logger.error(f"Error creating SHAP visualizations: {str(e)}")
        logger.error("If using NumPy 2.0+, you may need to update the SHAP library for compatibility")
    
    return figures


#--------------- Outlier Processing Functions ---------------#

def process_outliers_with_mean(df, outlier_flags_prefix, columns=None, output_prefix='processed_'):
    """
    Process outliers by replacing them with column means.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    outlier_flags_prefix : str
        Prefix of columns containing outlier flags (e.g., 'iqr_out_', 'zscore_out_')
    columns : list, default=None
        List of column names to process. If None, uses all columns with corresponding flag columns
    output_prefix : str, default='processed_'
        Prefix for the new processed columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added processed columns
    """
    df_copy = df.copy()
    
    # If columns not specified, find all columns with corresponding flag columns
    if columns is None:
        flag_columns = [col for col in df.columns if col.startswith(outlier_flags_prefix)]
        columns = [col.replace(outlier_flags_prefix, '') for col in flag_columns]
    
    # Process each column
    for col in columns:
        flag_col = f"{outlier_flags_prefix}{col}"
        
        # Skip if flag column doesn't exist
        if flag_col not in df.columns:
            logger.warning(f"Flag column {flag_col} not found, skipping {col}")
            continue
        
        # Calculate mean (excluding outliers for better estimate)
        mean_value = df[col][df[flag_col] == 0].mean()
        
        # Create processed column
        processed_col = f"{output_prefix}{col}"
        df_copy[processed_col] = df[col].copy()
        
        # Replace outliers with mean
        df_copy.loc[df[flag_col] == 1, processed_col] = mean_value
        
        logger.info(f"Processed column {col} by replacing {df[flag_col].sum()} outliers with mean value {mean_value:.4f}")
    
    return df_copy


def process_outliers_with_median(df, outlier_flags_prefix, columns=None, output_prefix='processed_'):
    """
    Process outliers by replacing them with column medians.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    outlier_flags_prefix : str
        Prefix of columns containing outlier flags (e.g., 'iqr_out_', 'zscore_out_')
    columns : list, default=None
        List of column names to process. If None, uses all columns with corresponding flag columns
    output_prefix : str, default='processed_'
        Prefix for the new processed columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added processed columns
    """
    df_copy = df.copy()
    
    # If columns not specified, find all columns with corresponding flag columns
    if columns is None:
        flag_columns = [col for col in df.columns if col.startswith(outlier_flags_prefix)]
        columns = [col.replace(outlier_flags_prefix, '') for col in flag_columns]
    
    # Process each column
    for col in columns:
        flag_col = f"{outlier_flags_prefix}{col}"
        
        # Skip if flag column doesn't exist
        if flag_col not in df.columns:
            logger.warning(f"Flag column {flag_col} not found, skipping {col}")
            continue
        
        # Calculate median (excluding outliers for better estimate)
        median_value = df[col][df[flag_col] == 0].median()
        
        # Create processed column
        processed_col = f"{output_prefix}{col}"
        df_copy[processed_col] = df[col].copy()
        
        # Replace outliers with median
        df_copy.loc[df[flag_col] == 1, processed_col] = median_value
        
        logger.info(f"Processed column {col} by replacing {df[flag_col].sum()} outliers with median value {median_value:.4f}")
    
    return df_copy


def process_outliers_with_winsorization(df, outlier_flags_prefix=None, columns=None, output_prefix='winsorized_', limits=(0.05, 0.95)):
    """
    Process outliers by winsorization (capping values at percentiles).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    outlier_flags_prefix : str, default=None
        Prefix of columns containing outlier flags. If None, applies winsorization to all values
    columns : list, default=None
        List of column names to process. If None, uses all numerical columns
    output_prefix : str, default='winsorized_'
        Prefix for the new processed columns
    limits : tuple, default=(0.05, 0.95)
        Tuple of (lower percentile, upper percentile) to use for capping
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added winsorized columns
    """
    try:
        from scipy.stats import mstats
    except ImportError:
        logger.error("scipy is required for winsorization")
        raise ImportError("scipy is required for winsorization")
    
    df_copy = df.copy()
    
    # If columns not specified, use all numerical columns
    if columns is None:
        columns = identify_numerical_columns(df)
    
    # Process each column
    for col in columns:
        # Only process if column exists
        if col not in df.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue
        
        # Create winsorized column
        winsorized_col = f"{output_prefix}{col}"
        
        # Apply winsorization
        if outlier_flags_prefix:
            # Use outlier flags to selectively winsorize
            flag_col = f"{outlier_flags_prefix}{col}"
            
            if flag_col not in df.columns:
                logger.warning(f"Flag column {flag_col} not found, skipping {col}")
                continue
            
            # Copy original column
            df_copy[winsorized_col] = df[col].copy()
            
            # Apply winsorization only to outliers
            outliers = df_copy[df[flag_col] == 1][col]
            if not outliers.empty:
                lower_bound = df[col][df[flag_col] == 0].quantile(limits[0])
                upper_bound = df[col][df[flag_col] == 0].quantile(limits[1])
                
                # Apply bounds to outliers
                df_copy.loc[df[flag_col] == 1, winsorized_col] = df_copy.loc[df[flag_col] == 1, col].clip(lower=lower_bound, upper=upper_bound)
                
                logger.info(f"Winsorized {df[flag_col].sum()} outliers in column {col}")
        else:
            # Winsorize entire column
            df_copy[winsorized_col] = mstats.winsorize(df[col].values, limits=limits)
            logger.info(f"Winsorized column {col} with limits {limits}")
    
    return df_copy


def process_outliers_with_trimming(df, outlier_flags_prefix, columns=None):
    """
    Process outliers by removing rows containing outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    outlier_flags_prefix : str
        Prefix of columns containing outlier flags (e.g., 'iqr_out_', 'zscore_out_')
    columns : list, default=None
        List of column names to consider for trimming. If None, uses all columns with flag columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with rows containing outliers removed
    """
    # If columns not specified, find all columns with corresponding flag columns
    if columns is None:
        flag_columns = [col for col in df.columns if col.startswith(outlier_flags_prefix)]
        columns = [col.replace(outlier_flags_prefix, '') for col in flag_columns]
    
    # Get flag columns
    flag_cols = [f"{outlier_flags_prefix}{col}" for col in columns if f"{outlier_flags_prefix}{col}" in df.columns]
    
    if not flag_cols:
        logger.warning(f"No flag columns with prefix {outlier_flags_prefix} found")
        return df.copy()
    
    # Create mask for rows to keep (where no column has outliers)
    mask = df[flag_cols].sum(axis=1) == 0
    
    # Apply mask to filter dataframe
    df_filtered = df[mask].copy()
    
    num_removed = len(df) - len(df_filtered)
    logger.info(f"Removed {num_removed} rows ({num_removed/len(df)*100:.2f}%) containing outliers")
    
    return df_filtered


#--------------- Main Analysis Function ---------------#

def analyze_outliers(df, columns=None, iqr_threshold=1.5, zscore_threshold=3, 
                     add_isolation_forest=True, tune_isolation_forest=False, contamination=0.05, 
                     save_path=None, dpi=300, exclude_target=True, use_optuna=True, n_trials=50, 
                     timeout=600):
    """
    Complete workflow for detecting outliers using multiple methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list, default=None
        List of column names to check for outliers. If None, uses all numerical columns
    iqr_threshold : float, default=1.5
        Multiplier for IQR to determine outlier boundaries
    zscore_threshold : float, default=3
        Z-score threshold beyond which a value is considered an outlier
    add_isolation_forest : bool, default=True
        Whether to include Isolation Forest outlier detection
    tune_isolation_forest : bool, default=False
        Whether to tune the Isolation Forest parameters
    contamination : float, default=0.05
        Expected proportion of outliers for Isolation Forest
    save_path : str, default=None
        Directory path to save results and visualizations. If None, results are only returned
    dpi : int, default=300
        DPI for saved visualizations
    
    Returns:
    --------
    tuple
        (df_with_flags, outlier_results)
        - df_with_flags: DataFrame with added outlier flag columns
        - outlier_results: Dict with detailed outlier information
    """
    if columns is None:
        columns = identify_numerical_columns(df)
        
    # Exclude target variable (esg_score) from outlier analysis if requested
    if exclude_target and 'esg_score' in columns:
        columns = [col for col in columns if col != 'esg_score']
        logger.info("Excluded esg_score from outlier analysis as it is the target variable")
    
    logger.info(f"Analyzing outliers in {len(columns)} numerical columns")
    
    # 1. Visualize distributions with boxplots
    if save_path:
        boxplot_path = os.path.join(save_path, 'boxplots')
    else:
        boxplot_path = None
    
    plot_boxplots(df, columns, save_path=boxplot_path, dpi=dpi)
    
    # 2. Detect outliers using IQR method
    logger.info("Detecting outliers using IQR method")
    iqr_flags, iqr_counts, iqr_details = detect_iqr_outliers(df, columns)
    
    # 3. Detect outliers using Z-score method
    logger.info("Detecting outliers using Z-score method")
    zscore_flags, zscore_counts, zscore_details = detect_zscore_outliers(df, columns, threshold=zscore_threshold)
    
    # 4. Add outlier flags to the DataFrame
    df_with_flags = add_iqr_outlier_flags(df, columns)
    df_with_flags = add_zscore_outlier_flags(df_with_flags, columns, threshold=zscore_threshold)
    
    # 5. Add Isolation Forest outliers if requested
    forest_model = None
    tuned_model = None
    forest_counts = None
    forest_tuned_counts = None
    
    if add_isolation_forest:
        try:
            logger.info("Detecting outliers using Isolation Forest")
            df_with_flags, forest_model = detect_isolation_forest_outliers(
                df_with_flags, 
                contamination=contamination, 
                features=columns
            )
            
            # Count isolation forest outliers
            forest_counts = pd.Series(
                {col: df_with_flags[df_with_flags['forest_outlier'] == 1][col].count() for col in columns},
                name='Isolation Forest Outliers'
            )
            
            # Add tuned Isolation Forest if requested
            if tune_isolation_forest:
                logger.info("Detecting outliers using tuned Isolation Forest")
                if save_path:
                    params_path = save_path
                else:
                    params_path = None
                    
                df_with_flags, best_params, tuned_model = detect_isolation_forest_tuned_outliers(
                    df_with_flags,
                    features=columns,
                    save_path=params_path,
                    use_optuna=use_optuna,
                    n_trials=n_trials,
                    timeout=timeout
                )
                
                # Count tuned isolation forest outliers
                forest_tuned_counts = pd.Series(
                    {col: df_with_flags[df_with_flags['forest_out_tuned_'] == 1][col].count() for col in columns},
                    name='Tuned Isolation Forest Outliers'
                )
                
                # Create comparison plot for original vs tuned Isolation Forest
                if save_path:
                    forest_comp_path = os.path.join(save_path, 'isolation_forest_comparison.png')
                    plot_isolation_forest_outliers(
                        df_with_flags,
                        original_column='forest_outlier',
                        tuned_column='forest_out_tuned_',
                        save_path=forest_comp_path,
                        dpi=dpi
                    )
            
            # Generate SHAP plots if forest model is available
            if forest_model is not None and save_path is not None:
                if isinstance(df_with_flags, pd.DataFrame):
                    X = df_with_flags[columns]
                    feature_names = columns
                else:
                    X = columns
                    feature_names = None
                
                shap_path = os.path.join(save_path, 'shap')
                os.makedirs(shap_path, exist_ok=True)
                
                try:
                    # Check for NumPy 2.0 compatibility
                    if hasattr(np, 'version') and int(np.version.version.split('.')[0]) >= 2:
                        logger.info("NumPy 2.0+ detected, using compatibility mode for SHAP")
                    
                    plot_shap_values(
                        forest_model,
                        X,
                        feature_names=feature_names,
                        max_display=min(30, len(columns)),
                        save_path=os.path.join(shap_path, 'tuned_model'),
                        dpi=dpi
                    )
                except Exception as e:
                    logger.warning(f"Error generating SHAP plots: {e}")
                    logger.warning("If using NumPy 2.0+, SHAP library may need updating for full compatibility")
        except ImportError:
            logger.warning("Skipping Isolation Forest detection (scikit-learn required)")
    
    # 6. Compare outlier detection methods
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        comparison_path = os.path.join(save_path, 'outliers_comparison.png')
    else:
        comparison_path = None
    
    plot_outlier_comparison(
        iqr_counts, 
        zscore_counts, 
        forest_outliers=forest_counts, 
        save_path=comparison_path, 
        dpi=dpi,
        include_isolation_forest=False  # Set to False to exclude Isolation Forest outliers
    )
    
    # 7. Compile and save results
    outlier_results = {
        'iqr': {
            'flags': iqr_flags,
            'counts': iqr_counts,
            'details': iqr_details
        },
        'zscore': {
            'flags': zscore_flags,
            'counts': zscore_counts,
            'details': zscore_details
        }
    }
    
    # Add isolation forest results if available
    if forest_counts is not None:
        outlier_results['isolation_forest'] = {
            'counts': forest_counts,
            'model': forest_model
        }
        
        if forest_tuned_counts is not None:
            outlier_results['isolation_forest_tuned'] = {
                'counts': forest_tuned_counts,
                'model': tuned_model
            }
    
    if save_path:
        # Save outlier counts to CSV
        counts_dict = {
            'IQR_Outliers': iqr_counts,
            'ZScore_Outliers': zscore_counts
        }
        
        if forest_counts is not None:
            counts_dict['Forest_Outliers'] = forest_counts
            
        if forest_tuned_counts is not None:
            counts_dict['Tuned_Forest_Outliers'] = forest_tuned_counts
            
        counts_df = pd.DataFrame(counts_dict)
        counts_path = os.path.join(save_path, 'outlier_counts.csv')
        counts_df.to_csv(counts_path)
        logger.info(f"Outlier counts saved to {counts_path}")
    
    logger.info(f"Outlier analysis complete. Found {iqr_counts.sum()} IQR outliers and {zscore_counts.sum()} Z-score outliers")
    
    return df_with_flags, outlier_results


def process_and_evaluate_outliers(df_with_flags, method='mean', outlier_flags_prefix='iqr_out_',
                                  columns=None, save_path=None, dpi=300):
    """
    Process outliers using the specified method and evaluate the results.
    
    Parameters:
    -----------
    df_with_flags : pandas.DataFrame
        DataFrame with outlier flag columns
    method : str, default='mean'
        Method to use for outlier processing: 'mean', 'median', 'winsorize', or 'trim'
    outlier_flags_prefix : str, default='iqr_out_'
        Prefix of columns containing outlier flags
    columns : list, default=None
        List of column names to process. If None, uses all columns with flag columns
    save_path : str, default=None
        Directory to save visualizations. If None, plots are only displayed
    dpi : int, default=300
        DPI for saved plots
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with processed values
    """
    # Select processing method
    if method.lower() == 'mean':
        logger.info("Processing outliers by replacing with column means")
        processed_df = process_outliers_with_mean(df_with_flags, outlier_flags_prefix, columns)
        prefix = 'processed_'
    elif method.lower() == 'median':
        logger.info("Processing outliers by replacing with column medians")
        processed_df = process_outliers_with_median(df_with_flags, outlier_flags_prefix, columns)
        prefix = 'processed_'
    elif method.lower() == 'winsorize':
        logger.info("Processing outliers by winsorization")
        processed_df = process_outliers_with_winsorization(df_with_flags, outlier_flags_prefix, columns)
        prefix = 'winsorized_'
    elif method.lower() == 'trim':
        logger.info("Processing outliers by trimming (removing outlier rows)")
        processed_df = process_outliers_with_trimming(df_with_flags, outlier_flags_prefix, columns)
        return processed_df  # No visualization for trimming
    else:
        logger.error(f"Unknown processing method: {method}")
        raise ValueError(f"Unknown processing method: {method}")
    
    # If columns not specified, find all columns with corresponding flag columns
    if columns is None:
        flag_columns = [col for col in df_with_flags.columns if col.startswith(outlier_flags_prefix)]
        columns = [col.replace(outlier_flags_prefix, '') for col in flag_columns]
    
    # No longer generate individual boxplot comparisons
    # The comparison plots are now generated via the notebook using a multi-plot approach
    
    return processed_df


def visualize_outlier_processing(original_df, processed_df, columns, prefix='processed_', save_path=None, dpi=300):
    """
    Visualize the effect of outlier processing by comparing original and processed distributions.
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        Original DataFrame with outliers
    processed_df : pandas.DataFrame
        Processed DataFrame with handled outliers
    columns : list
        List of column names to visualize
    prefix : str, default='processed_'
        Prefix for processed columns in processed_df
    save_path : str, default=None
        Directory to save visualizations. If None, plots are only displayed
    dpi : int, default=300
        DPI for saved plots
        
    Returns:
    --------
    list
        List of created figure objects
    """
    figures = []
    
    for col in columns:
        processed_col = f"{prefix}{col}"
        
        if processed_col not in processed_df.columns:
            logger.warning(f"Processed column {processed_col} not found, skipping visualization")
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        figures.append(fig)
        
        # Original data boxplot
        sns.boxplot(y=col, data=original_df, ax=axes[0])
        axes[0].set_title(f"Original: {col}")
        
        # Processed data boxplot
        sns.boxplot(y=processed_col, data=processed_df, ax=axes[1])
        axes[1].set_title(f"Processed: {processed_col}")
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"outlier_processing_{col}.png")
            plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Visualization saved to {file_path}")
    
    return figures


def plot_multi_boxplot_comparisons(df, original_cols, modified_cols_prefix, 
                                 num_cols=3, plots_per_figure=3, 
                                 save_dir=None, dpi=300, fig_name_prefix=None):
    """
    Create multi-boxplot comparison visualizations showing original and 
    modified distributions side by side.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing both original and modified columns
    original_cols : list
        List of original column names
    modified_cols_prefix : str
        Prefix for modified columns (e.g., 'iqr_mod_' or 'z_mod_')
    num_cols : int, default=3
        Number of columns in each figure
    plots_per_figure : int, default=3
        Number of rows per figure
    save_dir : str, default=None
        Directory to save plots. If None, plots are only displayed
    dpi : int, default=300
        DPI for saved plots
    fig_name_prefix : str, default=None
        Prefix for the figure filenames. If None, uses 'comparison_boxplots'
        
    Returns:
    --------
    list
        List of created figure objects
    """
    import math
    figures = []
    
    # Calculate the total number of rows and figures needed
    num_rows = math.ceil(len(original_cols) / num_cols)
    total_plots = math.ceil(num_rows / plots_per_figure)
    
    # Default figure name prefix
    if fig_name_prefix is None:
        if 'iqr' in modified_cols_prefix.lower():
            fig_name_prefix = 'comparison_boxplots_iqr_handling'
        elif 'z' in modified_cols_prefix.lower():
            fig_name_prefix = 'comparison_boxplots_z_handling'
        else:
            fig_name_prefix = 'comparison_boxplots'
    
    logger.info(f"Generating {total_plots} multi-boxplot comparison figures")
    
    for plot_num in range(total_plots):
        start_idx = plot_num * plots_per_figure * num_cols
        end_idx = min((plot_num + 1) * plots_per_figure * num_cols, len(original_cols))
        
        current_features = original_cols[start_idx:end_idx]
        current_num_rows = math.ceil(len(current_features) / num_cols)

        fig, axes = plt.subplots(current_num_rows, num_cols, figsize=(12, 4 * current_num_rows))  # Increased figure size
        figures.append(fig)
        
        # Convert axes to array if it's a single subplot
        if current_num_rows * num_cols == 1:
            axes = np.array([axes])
        
        # Flatten the axes array for easier indexing
        axes = axes.flatten()
                    
        for i, col in enumerate(current_features):
            modified_col = f"{modified_cols_prefix}{col}"
            
            # Check if the column exists in the dataframe
            if modified_col in df.columns:
                # Plot the boxplot comparison
                axes[i].boxplot(
                    [
                        df[col].dropna(),               # Original column values
                        df[modified_col].dropna()       # Modified column values
                    ],
                    labels=[col, modified_col],         # Labels for the boxplots
                    medianprops=dict(color="blue", linewidth=1.5)  # Style median line
                )
                axes[i].set_title(
                    f"Comparison of {col} \nand {modified_col}", 
                    fontsize=12,                      # Increased title font size
                    fontweight='bold'                  # Bold title
                )
                axes[i].set_ylabel('', fontsize=10)    # Increased label font size
                # Adjust tick label font size
                axes[i].tick_params(axis='x', labelsize=10)  # Increased tick label size
                axes[i].tick_params(axis='y', labelsize=10)  # Increased tick label size
            else:
                # Handle the case where the modified column is not found
                axes[i].set_title(f"{modified_col} not found", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the figure if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_name = f"{fig_name_prefix}_part_{plot_num + 1}.png"
            full_path = os.path.join(save_dir, file_name)
            plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Multi-boxplot comparison saved to {full_path}")
    
    return figures