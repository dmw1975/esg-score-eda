#!/usr/bin/env python3
"""
Feature Engineering Module for ESG Score Data Analysis

This module provides functions for feature transformation, including
one-hot encoding and other feature engineering techniques.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


def identify_categorical_columns(df, exclude_columns=None):
    """
    Identify categorical columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    exclude_columns : list, default=None
        List of column names to exclude from consideration
        
    Returns:
    --------
    list
        Column names of categorical variables
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Identify object and category dtypes
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    
    # Remove excluded columns
    categorical_cols = [col for col in categorical_cols if col not in exclude_columns]
    
    return categorical_cols


def one_hot_encode(df, categorical_columns=None, exclude_columns=None, 
                   drop_first=True, dtype='float32'):
    """
    Perform one-hot encoding on categorical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    categorical_columns : list, default=None
        List of column names to encode. If None, automatically identifies categorical columns
    exclude_columns : list, default=None
        List of column names to exclude from encoding
    drop_first : bool, default=True
        Whether to drop the first category for each feature to avoid multicollinearity
    dtype : str, default='float32'
        Data type for encoded columns
        
    Returns:
    --------
    tuple
        (encoded_df, feature_mapping)
        - encoded_df: DataFrame with categorical columns encoded
        - feature_mapping: Dictionary mapping original columns to encoded columns
    """
    df_copy = df.copy()
    
    # Identify categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = identify_categorical_columns(df_copy, exclude_columns)
    
    logger.info(f"One-hot encoding {len(categorical_columns)} categorical columns")
    
    # Create a mapping to track original to encoded columns
    feature_mapping = {}
    
    # Process each categorical column separately to track the mapping
    all_encoded_columns = []
    
    for col in categorical_columns:
        # Get dummies for current column
        encoded = pd.get_dummies(df_copy[col], prefix=col, drop_first=drop_first)
        
        # Store mapping
        feature_mapping[col] = list(encoded.columns)
        
        # Convert to specified dtype
        encoded = encoded.astype(dtype)
        
        # Add to list of all encoded columns
        all_encoded_columns.append(encoded)
    
    # Combine all encoded columns
    encoded_cols = pd.concat(all_encoded_columns, axis=1)
    
    # Drop original categorical columns
    df_copy = df_copy.drop(columns=categorical_columns)
    
    # Combine with encoded columns
    encoded_df = pd.concat([df_copy, encoded_cols], axis=1)
    
    logger.info(f"Expanded {len(categorical_columns)} columns into {len(encoded_cols.columns)} binary features")
    
    return encoded_df, feature_mapping


def create_feature_expansion_table(df, feature_mapping, save_path=None):
    """
    Create a table documenting how categorical features are expanded.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The original dataframe with categorical columns
    feature_mapping : dict
        Dictionary mapping original columns to encoded columns
    save_path : str, default=None
        Path to save the documentation. If None, only returns the table
        
    Returns:
    --------
    pandas.DataFrame
        Table with feature expansion information
    """
    # Create a list to store information
    expansion_data = []
    
    for original_col, encoded_cols in feature_mapping.items():
        # Count unique values in original column
        n_categories = df[original_col].nunique()
        
        # Count binary features created
        n_features = len(encoded_cols)
        
        # Calculate expansion ratio
        expansion_ratio = n_features / n_categories
        
        # Record information
        expansion_data.append({
            'Original Column': original_col,
            'Unique Categories': n_categories,
            'Binary Features': n_features,
            'Expansion Ratio': expansion_ratio
        })
    
    # Create DataFrame
    expansion_table = pd.DataFrame(expansion_data)
    
    # Add totals
    total_categories = expansion_table['Unique Categories'].sum()
    total_features = expansion_table['Binary Features'].sum()
    total_ratio = total_features / total_categories if total_categories > 0 else 0
    
    # Create a row for totals
    totals = pd.DataFrame([{
        'Original Column': 'TOTAL',
        'Unique Categories': total_categories,
        'Binary Features': total_features,
        'Expansion Ratio': total_ratio
    }])
    
    # Combine
    expansion_table = pd.concat([expansion_table, totals])
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as CSV
        expansion_table.to_csv(save_path, index=False)
        logger.info(f"Feature expansion table saved to {save_path}")
        
        # Save as LaTeX if requested (if path ends with .tex)
        if save_path.endswith('.tex'):
            latex_path = save_path
        else:
            latex_path = os.path.splitext(save_path)[0] + '.tex'
        
        expansion_table.to_latex(latex_path, index=False, float_format=lambda x: f"{x:.2f}")
        logger.info(f"Feature expansion table (LaTeX) saved to {latex_path}")
    
    return expansion_table


def create_feature_expansion_visualizations(expansion_table, save_path=None, dpi=300):
    """
    Create visualizations of feature expansion from one-hot encoding.
    
    Parameters:
    -----------
    expansion_table : pandas.DataFrame
        Table with feature expansion information
    save_path : str, default=None
        Directory to save visualizations. If None, only displays plots
    dpi : int, default=300
        DPI for saved visualizations
        
    Returns:
    --------
    list
        List of created figure objects
    """
    figures = []
    
    # Filter out the "TOTAL" row
    data = expansion_table[expansion_table['Original Column'] != 'TOTAL'].copy()
    
    # Sort by number of binary features
    data = data.sort_values('Binary Features', ascending=False)
    
    # 1. Bar chart of binary features by column
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    figures.append(fig1)
    
    bars = ax1.bar(data['Original Column'], data['Binary Features'])
    ax1.set_xlabel('Original Categorical Column')
    ax1.set_ylabel('Number of Binary Features Created')
    ax1.set_title('Binary Features Created by One-Hot Encoding')
    ax1.set_xticklabels(data['Original Column'], rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'binary_features_count.png'), dpi=dpi, bbox_inches='tight')
    
    # 2. Bar chart of original categories vs. binary features
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    figures.append(fig2)
    
    x = np.arange(len(data))
    width = 0.35
    
    ax2.bar(x - width/2, data['Unique Categories'], width, label='Unique Categories', color='#3498db')
    ax2.bar(x + width/2, data['Binary Features'], width, label='Binary Features', color='#e74c3c')
    
    ax2.set_xlabel('Original Categorical Column', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Comparison of Original Categories vs. Binary Features', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(data['Original Column'], rotation=45, ha='right')
    ax2.legend()
    
    # Add value labels
    bars1 = ax2.patches[:len(data)]
    bars2 = ax2.patches[len(data):]
    
    for bar in bars1 + bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{int(bar.get_height())}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'categories_vs_features.png'), dpi=dpi, bbox_inches='tight')
    
    # 3. Bar chart of expansion ratio
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    figures.append(fig3)
    
    # Create a copy for expansion ratio calculation
    expansion_ratio = data.copy()
    expansion_ratio['Expansion Ratio'] = expansion_ratio['Binary Features'] / expansion_ratio['Unique Categories']
    
    # Sort by expansion ratio
    expansion_ratio = expansion_ratio.sort_values('Expansion Ratio', ascending=False)
    
    # Create horizontal bar chart for better visualization
    bars = ax3.barh(expansion_ratio['Original Column'], expansion_ratio['Expansion Ratio'], color='#2ecc71')
    ax3.set_ylabel('Original Categorical Column', fontsize=12)
    ax3.set_xlabel('Expansion Ratio (Binary Features / Categories)', fontsize=12)
    ax3.set_title('Feature Expansion Ratio from One-Hot Encoding', fontsize=14)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}', ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'expansion_ratio.png'), dpi=dpi, bbox_inches='tight')
    
    # 4. Top features visualization
    if len(data) > 5:  # Only if we have enough features
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        figures.append(fig4)
        
        # Select top N variables by encoded features for better visualization
        top_n = min(10, len(data))
        top_vars = data.nlargest(top_n, 'Binary Features')
        
        x = np.arange(len(top_vars))
        width = 0.35
        
        ax4.bar(x - width/2, top_vars['Unique Categories'], width, label='Original Categories', color='#3498db')
        ax4.bar(x + width/2, top_vars['Binary Features'], width, label='Binary Features', color='#e74c3c')
        
        ax4.set_xlabel('Categorical Variables', fontsize=12)
        ax4.set_ylabel('Number of Features', fontsize=12)
        ax4.set_title('Top Variables by Feature Expansion', fontsize=14)
        ax4.set_xticks(x)
        ax4.set_xticklabels(top_vars['Original Column'], rotation=45, ha='right', fontsize=10)
        ax4.legend()
        
        # Add labels
        bars1 = ax4.patches[:len(top_vars)]
        bars2 = ax4.patches[len(top_vars):]
        
        for bar in bars1 + bars2:
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{int(bar.get_height())}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'feature_expansion_top.png'), dpi=dpi, bbox_inches='tight')
    
    return figures


def normalize_features(df, columns=None, method='standard', exclude_columns=None):
    """
    Normalize numerical features using specified method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list, default=None
        List of column names to normalize. If None, uses all numerical columns
    method : str, default='standard'
        Normalization method: 'standard', 'minmax', or 'robust'
    exclude_columns : list, default=None
        List of column names to exclude from normalization
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with normalized features
    """
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    except ImportError:
        logger.error("scikit-learn is required for normalization")
        raise ImportError("scikit-learn is required for normalization")
    
    df_copy = df.copy()
    
    # Identify columns to normalize if not provided
    if columns is None:
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        
        # Exclude specified columns
        if exclude_columns:
            columns = [col for col in numerical_cols if col not in exclude_columns]
        else:
            columns = numerical_cols
    
    # Select appropriate scaler
    if method.lower() == 'standard':
        scaler = StandardScaler()
    elif method.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif method.lower() == 'robust':
        scaler = RobustScaler()
    else:
        logger.error(f"Unknown normalization method: {method}")
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Apply scaling
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    
    logger.info(f"Normalized {len(columns)} features using {method} scaling")
    
    return df_copy


def target_encoding(df, categorical_columns, target_column, k=5, method='mean'):
    """
    Perform target encoding on categorical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    categorical_columns : list
        List of categorical column names to encode
    target_column : str
        Name of the target variable
    k : int, default=5
        Smoothing factor for regularization
    method : str, default='mean'
        Encoding method: 'mean' or 'median'
        
    Returns:
    --------
    tuple
        (encoded_df, encoding_dict)
        - encoded_df: DataFrame with target-encoded features
        - encoding_dict: Dictionary with encoding mappings
    """
    df_copy = df.copy()
    global_target = df[target_column].mean() if method == 'mean' else df[target_column].median()
    encoding_dict = {}
    
    for col in categorical_columns:
        # Calculate target statistics by category
        if method == 'mean':
            stats = df.groupby(col)[target_column].agg(['mean', 'count'])
            encoding_dict[col] = stats.copy()
            
            # Apply smoothing
            smoothed_mean = (stats['count'] * stats['mean'] + k * global_target) / (stats['count'] + k)
            encoding_dict[col]['smoothed'] = smoothed_mean
            
            # Map values back to the dataframe
            df_copy[f'{col}_target_encoded'] = df_copy[col].map(smoothed_mean)
        else:
            stats = df.groupby(col)[target_column].agg(['median', 'count'])
            encoding_dict[col] = stats.copy()
            
            # No smoothing for median
            df_copy[f'{col}_target_encoded'] = df_copy[col].map(stats['median'])
    
    logger.info(f"Applied target encoding to {len(categorical_columns)} categorical features")
    
    return df_copy, encoding_dict


def create_interaction_features(df, features_to_interact, interaction_type='multiply'):
    """
    Create interaction features between numerical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    features_to_interact : list of tuples
        List of tuples containing pairs of column names to interact
    interaction_type : str, default='multiply'
        Type of interaction: 'multiply', 'add', 'subtract', or 'divide'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added interaction features
    """
    df_copy = df.copy()
    
    for feat1, feat2 in features_to_interact:
        if feat1 not in df.columns or feat2 not in df.columns:
            logger.warning(f"Features {feat1} or {feat2} not found in DataFrame, skipping")
            continue
        
        if interaction_type == 'multiply':
            df_copy[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        elif interaction_type == 'add':
            df_copy[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
        elif interaction_type == 'subtract':
            df_copy[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
        elif interaction_type == 'divide':
            # Handle division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                df_copy[f'{feat1}_div_{feat2}'] = df[feat1] / df[feat2]
                df_copy[f'{feat1}_div_{feat2}'].replace([np.inf, -np.inf], np.nan, inplace=True)
        else:
            logger.error(f"Unknown interaction type: {interaction_type}")
            raise ValueError(f"Unknown interaction type: {interaction_type}")
    
    logger.info(f"Created {len(features_to_interact)} interaction features using {interaction_type}")
    
    return df_copy


def polynomial_features(df, columns, degree=2):
    """
    Generate polynomial features from numerical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list
        List of column names to use for polynomial feature generation
    degree : int, default=2
        Degree of the polynomial features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added polynomial features
    """
    try:
        from sklearn.preprocessing import PolynomialFeatures
    except ImportError:
        logger.error("scikit-learn is required for polynomial features")
        raise ImportError("scikit-learn is required for polynomial features")
    
    df_copy = df.copy()
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[columns])
    
    # Get feature names
    feature_names = poly.get_feature_names_out(columns)
    
    # Add polynomial features to dataframe
    for i, name in enumerate(feature_names):
        if name in columns:  # Skip original features
            continue
        # Clean up feature names for clarity
        clean_name = name.replace(' ', '_').replace('^', '_pow_')
        df_copy[clean_name] = poly_features[:, i]
    
    logger.info(f"Generated polynomial features of degree {degree} from {len(columns)} features")
    
    return df_copy


def engineer_features(df, categorical_cols=None, exclude_cols=None, one_hot=True, normalization=None, 
                      interactions=None, polynomials=None, poly_degree=2, save_path=None, dpi=300):
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    categorical_cols : list, default=None
        List of categorical column names. If None, auto-detected
    exclude_cols : list, default=None
        List of column names to exclude from processing
    one_hot : bool, default=True
        Whether to apply one-hot encoding to categorical variables
    normalization : str, default=None
        Normalization method: 'standard', 'minmax', 'robust', or None
    interactions : list of tuples, default=None
        List of feature pairs to create interaction features
    polynomials : list, default=None
        List of column names to use for polynomial features
    poly_degree : int, default=2
        Degree of polynomial features
    save_path : str, default=None
        Directory to save results and visualizations
    dpi : int, default=300
        DPI for visualizations
        
    Returns:
    --------
    tuple
        (processed_df, feature_info)
        - processed_df: DataFrame with engineered features
        - feature_info: Dictionary with information about feature transformations
    """
    logger.info("Starting feature engineering pipeline")
    
    # Make a copy of the input dataframe
    processed_df = df.copy()
    
    # Track feature transformations
    feature_info = {
        'original_shape': df.shape,
        'transformations': []
    }
    
    # 1. Apply one-hot encoding if requested
    if one_hot:
        logger.info("Applying one-hot encoding")
        processed_df, feature_mapping = one_hot_encode(
            processed_df, 
            categorical_columns=categorical_cols,
            exclude_columns=exclude_cols
        )
        
        feature_info['transformations'].append({
            'type': 'one_hot_encoding',
            'feature_mapping': feature_mapping
        })
        
        # Create documentation if save path provided
        if save_path:
            doc_path = os.path.join(save_path, 'feature_expansion_table.csv')
            expansion_table = create_feature_expansion_table(df, feature_mapping, save_path=doc_path)
            
            # Create visualizations
            vis_path = os.path.join(save_path, 'visualizations')
            create_feature_expansion_visualizations(expansion_table, save_path=vis_path, dpi=dpi)
    
    # 2. Apply normalization if requested
    if normalization:
        logger.info(f"Normalizing numerical features using {normalization} method")
        
        # Determine columns to normalize
        normalize_cols = processed_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        
        if exclude_cols:
            normalize_cols = [col for col in normalize_cols if col not in exclude_cols]
        
        processed_df = normalize_features(
            processed_df,
            columns=normalize_cols,
            method=normalization
        )
        
        feature_info['transformations'].append({
            'type': 'normalization',
            'method': normalization,
            'columns': list(normalize_cols)
        })
    
    # 3. Create interaction features if requested
    if interactions:
        logger.info(f"Creating {len(interactions)} interaction features")
        processed_df = create_interaction_features(
            processed_df,
            features_to_interact=interactions
        )
        
        feature_info['transformations'].append({
            'type': 'interactions',
            'interactions': interactions
        })
    
    # 4. Generate polynomial features if requested
    if polynomials:
        logger.info(f"Generating polynomial features of degree {poly_degree}")
        processed_df = polynomial_features(
            processed_df,
            columns=polynomials,
            degree=poly_degree
        )
        
        feature_info['transformations'].append({
            'type': 'polynomials',
            'columns': polynomials,
            'degree': poly_degree
        })
    
    # Record final shape
    feature_info['final_shape'] = processed_df.shape
    feature_info['feature_count_change'] = processed_df.shape[1] - df.shape[1]
    
    logger.info(f"Feature engineering complete. Features increased from {df.shape[1]} to {processed_df.shape[1]}")
    
    # Create datatype summary if save path provided
    if save_path:
        # Create dtype summary
        dtype_path = os.path.join(save_path, 'dtype_summary')
        create_datatype_summary(processed_df, save_path=dtype_path, dpi=dpi)
        logger.info(f"Datatype summary created in {dtype_path}")
        
        # Create distribution plots for numerical features that exist in both dataframes
        common_numerical = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.intersection(
            processed_df.columns)
        
        if len(common_numerical) > 0:
            dist_path = os.path.join(save_path, 'distributions')
            plot_feature_distributions(df, processed_df, numerical_cols=common_numerical, 
                                      save_path=dist_path, dpi=dpi)
            logger.info(f"Feature distribution plots created in {dist_path}")
    
    return processed_df, feature_info


def create_datatype_summary(df, save_path=None, dpi=300):
    """
    Create summary visualizations of dataframe column types and memory usage.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    save_path : str, default=None
        Directory to save visualizations. If None, only displays plots
    dpi : int, default=300
        DPI for saved visualizations
        
    Returns:
    --------
    tuple
        (dtype_counts, memory_usage_df)
        - dtype_counts: Series with counts of each datatype
        - memory_usage_df: DataFrame with memory usage by column
    """
    # Count column datatypes
    dtype_counts = df.dtypes.value_counts()
    
    # Get memory usage by column
    memory_usage = df.memory_usage(deep=True)
    memory_usage_df = pd.DataFrame({
        'Column': memory_usage.index, 
        'Memory (bytes)': memory_usage.values
    })
    memory_usage_df = memory_usage_df.sort_values('Memory (bytes)', ascending=False)
    
    # Set up plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Datatype counts plot
    bars = axes[0].bar(dtype_counts.index.astype(str), dtype_counts.values, color='#3498db')
    axes[0].set_title('Column Datatypes Distribution', fontsize=14)
    axes[0].set_xlabel('Datatype', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 2. Memory usage plot (top 10 columns)
    top_n = min(10, len(memory_usage_df))
    top_memory = memory_usage_df.head(top_n)
    
    # Convert to MB for better display
    top_memory['Memory (MB)'] = top_memory['Memory (bytes)'] / (1024 * 1024)
    
    bars = axes[1].barh(top_memory['Column'], top_memory['Memory (MB)'], color='#e74c3c')
    axes[1].set_title('Memory Usage by Column (Top 10)', fontsize=14)
    axes[1].set_xlabel('Memory Usage (MB)', fontsize=12)
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f} MB', ha='left', va='center')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'datatype_summary.png'), dpi=dpi, bbox_inches='tight')
        
        # Also save datatypes as CSV
        dtype_df = pd.DataFrame({
            'Datatype': dtype_counts.index.astype(str),
            'Count': dtype_counts.values
        })
        dtype_df.to_csv(os.path.join(save_path, 'datatype_counts.csv'), index=False)
        
        # Save memory usage
        memory_usage_df.to_csv(os.path.join(save_path, 'memory_usage.csv'), index=False)
        logger.info(f"Datatype summary saved to {save_path}")
    
    return dtype_counts, memory_usage_df


def plot_feature_distributions(original_df, transformed_df, numerical_cols=None, n_cols=3, n_rows=None, 
                              figsize=(16, 12), save_path=None, dpi=300):
    """
    Plot the distribution of numerical features before and after transformation.
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        The original dataframe before transformation
    transformed_df : pandas.DataFrame
        The transformed dataframe
    numerical_cols : list, default=None
        List of numerical columns to plot. If None, selects numerical columns from original_df
    n_cols : int, default=3
        Number of columns in the subplot grid
    n_rows : int, default=None
        Number of rows in the subplot grid. If None, calculated based on n_cols and number of features
    figsize : tuple, default=(16, 12)
        Figure size (width, height) in inches
    save_path : str, default=None
        Directory to save visualizations. If None, only displays plots
    dpi : int, default=300
        DPI for saved visualizations
        
    Returns:
    --------
    list
        List of created figure objects
    """
    figures = []
    
    # Identify numerical columns if not provided
    if numerical_cols is None:
        numerical_cols = original_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    
    # Filter columns that exist in both dataframes
    numerical_cols = [col for col in numerical_cols if col in original_df.columns and col in transformed_df.columns]
    
    if not numerical_cols:
        logger.warning("No numerical columns found in both dataframes")
        return figures
    
    # Calculate number of rows if not provided
    if n_rows is None:
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    # Create subplots for each feature
    for i in range(0, len(numerical_cols), n_cols * n_rows):
        # Subset of features for this figure
        subset = numerical_cols[i:i + n_cols * n_rows]
        subset_rows = (len(subset) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(subset_rows, n_cols, figsize=figsize)
        figures.append(fig)
        
        # Flatten axes for easier indexing
        if subset_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif subset_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        
        for j, col in enumerate(subset):
            if j < len(axes):
                ax = axes[j]
                
                # Plot original distribution (blue)
                sns.histplot(original_df[col].dropna(), color='#3498db', alpha=0.5, label='Original', ax=ax)
                
                # Plot transformed distribution (red)
                sns.histplot(transformed_df[col].dropna(), color='#e74c3c', alpha=0.5, label='Transformed', ax=ax)
                
                ax.set_title(f'{col}', fontsize=11)
                ax.legend()
                
                # Add statistics
                orig_mean = original_df[col].mean()
                orig_std = original_df[col].std()
                trans_mean = transformed_df[col].mean()
                trans_std = transformed_df[col].std()
                
                stats_text = f'Orig: μ={orig_mean:.2f}, σ={orig_std:.2f}\nTrans: μ={trans_mean:.2f}, σ={trans_std:.2f}'
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                        verticalalignment='top', horizontalalignment='right',
                        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
        
        # Hide unused subplots
        for j in range(len(subset), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'feature_distributions_{i//n_cols//n_rows}.png'), 
                        dpi=dpi, bbox_inches='tight')
    
    return figures