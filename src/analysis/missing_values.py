#!/usr/bin/env python3
"""
Missing Values Analysis Module

This module provides functions for analyzing missing values in financial datasets,
particularly for ESG score analysis.
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import matplotlib.gridspec as gridspec
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Set up logging
logger = logging.getLogger(__name__)


def plot_missing_values(df, save_path=None, filename="missing_values.png", dpi=300):
    """
    Generates a bar chart displaying the percentage of missing values for each column.

    Parameters:
    df (pd.DataFrame): The input dataset.
    save_path (str, optional): Directory to save the plot. Defaults to None.
    filename (str, optional): Name of the saved plot file. Defaults to "missing_values.png".
    dpi (int, optional): Image resolution in dots per inch. Defaults to 300.

    Returns:
    None: Displays the plot and optionally saves it to the specified directory.
    """
    missing_percentage = df.isnull().mean() * 100
    missing_df = pd.DataFrame({"column": missing_percentage.index, "percent_missing": missing_percentage.values})
    missing_df = missing_df.sort_values("percent_missing", ascending=False)

    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    ax = sns.barplot(x="percent_missing", y="column", data=missing_df, color="firebrick")

    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if width > 0:
            ax.text(width + 0.5, p.get_y() + p.get_height()/2, f"{width:.1f}%", ha="left", va="center")

    plt.title("Percentage of Missing Values by Column", fontsize=16, pad=20)
    plt.xlabel("Percentage of Missing Values", fontsize=12)
    plt.ylabel("Column Names", fontsize=12)
    plt.xlim(0, max(missing_percentage.values) + 5 if len(missing_percentage) > 0 else 5)

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Figure saved to {full_path}")

    fig = plt.gcf()  # Get current figure
    plt.close()
    return fig


def plot_missingness_by_sector(df, numeric_columns, save_path=None, filename="missingness_by_sector.png", dpi=300):
    """
    Create visualizations of missing data patterns by sector.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'gics_sector' and other columns
    numeric_columns (list): List of numeric column names to check for missingness
    save_path (str, optional): Directory path to save the figure
    filename (str, optional): Filename for the saved figure
    dpi (int, optional): Resolution for the saved figure
    
    Returns:
    pandas.DataFrame: DataFrame of missingness percentages by sector
    """
    # Create a binary missing data indicator matrix (1 = missing, 0 = not missing)
    missing_matrix = df[numeric_columns].isnull().astype(int)
    
    # Add sector column
    missing_matrix['sector'] = df['gics_sector']
    
    # Calculate missingness percentage by sector
    sector_missing = {}
    for sector in missing_matrix['sector'].unique():
        sector_data = missing_matrix[missing_matrix['sector'] == sector]
        sector_missing[sector] = {col: sector_data[col].mean() * 100 for col in numeric_columns}
    
    # Convert to DataFrame for easier plotting
    sector_missing_df = pd.DataFrame(sector_missing).T
    
    # Sort columns by overall missingness
    overall_missing = missing_matrix[numeric_columns].mean().sort_values(ascending=False)
    top_missing_cols = overall_missing.index[:10]  # Top 10 columns with most missing values
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Heatmap of missingness by sector
    ax0 = plt.subplot(gs[0])
    sns.heatmap(
        sector_missing_df[top_missing_cols], 
        cmap="YlOrRd", 
        annot=True, 
        fmt=".1f",
        cbar_kws={'label': 'Missing Values (%)'},
        ax=ax0
    )
    ax0.set_title('Percentage of Missing Values by Sector (Top 10 Variables)', fontsize=14)
    ax0.set_ylabel('Sector', fontsize=12)
    
    # Save the figure if save_path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Construct full path
        full_path = os.path.join(save_path, filename)
        
        # Save the figure
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved to {full_path}")
        
        # Create and save CSV of top 10 missing variables
        # Extract the top 10 missing variables data
        top_missing_df = pd.DataFrame({
            'variable': top_missing_cols,
            'missing_percentage': overall_missing[:10].values
        })
        
        # Create CSV filename by replacing the extension of the image filename
        csv_filename = os.path.splitext(filename)[0] + '_top10_missing.csv'
        csv_path = os.path.join(save_path, csv_filename)
        
        # Save to CSV
        top_missing_df.to_csv(csv_path, index=False)
        logger.info(f"Top 10 missing variables saved to {csv_path}")
    
    # Return for use in the chi-square visualization
    return sector_missing_df


def test_sector_missingness_association(df, save_path=None, filename="chi_square_results.png", dpi=300, threshold=0.05):
    """
    Perform Chi-square test to check association between sector classification 
    and missing data patterns, focusing on sectors with HIGHER missing rates than average.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'gics_sector' and other columns
    save_path (str, optional): Directory path to save the figure
    filename (str, optional): Filename for the saved figure
    dpi (int, optional): Resolution for the saved figure
    threshold (float, optional): Threshold for difference in missing rates (default 5%)
    
    Returns:
    tuple: Chi-square statistic, p-value, degrees of freedom, expected frequencies
    """
    # Select numeric columns (excluding sector)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create a binary missing data indicator matrix (1 = missing, 0 = not missing)
    missing_matrix = df[numeric_columns].isnull().astype(int)
    
    # Add sector column to this matrix
    missing_matrix['sector'] = df['gics_sector']
    
    # Create contingency tables for each variable with sector
    results = {}
    
    logger.info("Chi-square test results for association between sector and missingness:")
    
    for col in numeric_columns:
        # Cross-tabulation of sector and missingness for this variable
        contingency_table = pd.crosstab(missing_matrix['sector'], missing_matrix[col])
        
        # Only perform test if there are both missing and non-missing values
        if contingency_table.shape[1] > 1:
            # Chi-square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            results[col] = {
                'chi2': chi2,
                'p_value': p,
                'dof': dof,
                'significant': p < 0.05
            }
            
            logger.info(f"Variable: {col}")
            logger.info(f"Chi-square: {chi2:.2f}, p-value: {p:.6f}, DoF: {dof}")
            logger.info(f"Association is {'statistically significant' if p < 0.05 else 'not significant'}")
    
    # Count significant associations
    sig_count = sum(1 for r in results.values() if r['significant'])
    
    # Dictionary to track significant variables by sector (with HIGHER missing rates)
    sectors = df['gics_sector'].unique()
    sector_sig_vars = {sector: [] for sector in sectors}
    sector_sig_counts = {}
    
    for sector in sectors:
        sector_data = missing_matrix[missing_matrix['sector'] == sector]
        sig_vars_for_sector = 0
        
        for col, result in results.items():
            if result['significant']:
                # Calculate missingness difference from overall average
                sector_miss_rate = sector_data[col].mean()
                overall_miss_rate = missing_matrix[col].mean()
                
                # MODIFICATION: Only consider cases where sector is missing MORE data than average
                # and the difference exceeds the threshold
                if (sector_miss_rate - overall_miss_rate) > threshold:
                    sig_vars_for_sector += 1
                    sector_sig_vars[sector].append({
                        'variable': col,
                        'sector_rate': sector_miss_rate,
                        'overall_rate': overall_miss_rate,
                        'difference': sector_miss_rate - overall_miss_rate,
                        'chi2': result['chi2'],
                        'p_value': result['p_value']
                    })
        
        sector_sig_counts[sector] = sig_vars_for_sector
    
    # Sort sectors by count of significant variables
    sorted_sectors = sorted(sector_sig_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_sector_names = [item[0] for item in sorted_sectors]
    sorted_sector_counts = [item[1] for item in sorted_sectors]
    
    # 1. Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_sector_names, sorted_sector_counts, color='firebrick')
    plt.title('Number of Variables with Significantly HIGHER Missing Data by Sector', fontsize=14)
    plt.xlabel('Sector', fontsize=12)
    plt.ylabel('Count of Significant Variables', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Save the bar chart if save_path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Construct full path for summary plot
        summary_path = os.path.join(save_path, "higher_missing_vars_count_by_sector.png")
        
        # Save the figure
        plt.savefig(summary_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Bar chart saved to {summary_path}")
    
    # 2. Create a separate table figure
    # Calculate how big the figure needs to be based on number of rows
    total_rows = sum(len(sector_sig_vars[sector]) for sector in sorted_sector_names if sector_sig_vars[sector])
    
    # Skip table creation if no significant higher missing rates found
    if total_rows == 0:
        logger.info("No sectors with significantly higher missing rates found.")
        return sum(r['chi2'] for r in results.values()), results
    
    # Adjust figure size based on number of rows
    fig_height = max(10, min(30, 4 + total_rows * 0.4))
    fig_width = 20
    
    fig_table = plt.figure(figsize=(fig_width, fig_height))
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('off')
    
    # Generate data for the table
    table_data = []
    row_labels = []
    col_labels = ['Variable', 'Sector Missing Rate', 'Overall Missing Rate', 'Difference', 'Chi-square', 'p-value']
    
    for sector in sorted_sector_names:
        if sector_sig_vars[sector]:
            # Sort variables by difference (highest first)
            sorted_vars = sorted(sector_sig_vars[sector], key=lambda x: x['difference'], reverse=True)
            
            # Format the data for this sector
            for var_data in sorted_vars:
                table_data.append([
                    var_data['variable'],
                    f"{var_data['sector_rate']:.1%}",
                    f"{var_data['overall_rate']:.1%}",
                    f"{var_data['difference']:+.1%}",
                    f"{var_data['chi2']:.2f}",
                    f"{var_data['p_value']:.4f}"
                ])
                row_labels.append(sector)
    
    # Create the table with improved styling
    table = ax_table.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.15, 0.15, 0.15, 0.12, 0.12]
    )
    
    # Style the table for better readability
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#B22222')  # Dark red for header (higher missing focus)
        
    # Style sector names (first column)
    for i in range(1, len(table_data) + 1):
        cell = table[i, -1]  # Row label cells
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#F5DEB3')  # Wheat color background
    
    # Style difference column - now all should be positive
    for row_idx in range(len(table_data)):
        # Difference column (column 3)
        cell = table[(row_idx + 1, 3)]
        # Color intensity based on magnitude of difference
        diff_value = float(cell.get_text().get_text().strip('%+'))
        intensity = min(255, 200 + int(diff_value * 5))  # Higher difference = more intense red
        cell.set_facecolor(f'#{intensity:02x}7070')  # Red with varying intensity
        
        # Alternate row colors for better readability
        for col in [0, 1, 2, 4, 5]:  # All columns except difference
            cell = table[(row_idx + 1, col)]
            if row_idx % 2 == 0:
                cell.set_facecolor('#F8F8F8')  # Very light gray for even rows
    
    plt.title(f'Variables with Significantly HIGHER Missing Data (>{threshold:.0%} above average)', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Calculate missing rate ranking for each sector (average rank across all variables)
    missing_ranks = {}
    for col in numeric_columns:
        # Get missing rates for each sector
        sector_rates = {}
        for sector in sectors:
            sector_data = missing_matrix[missing_matrix['sector'] == sector]
            sector_rates[sector] = sector_data[col].mean()
        
        # Rank sectors by missing rate (higher = worse rank)
        ranked_sectors = sorted(sector_rates.items(), key=lambda x: x[1], reverse=True)
        for rank, (sector, _) in enumerate(ranked_sectors):
            if sector not in missing_ranks:
                missing_ranks[sector] = []
            missing_ranks[sector].append(rank + 1)  # +1 so rank starts at 1 not 0
    
    # Calculate average rank
    avg_missing_ranks = {sector: np.mean(ranks) for sector, ranks in missing_ranks.items()}
    sorted_avg_ranks = sorted(avg_missing_ranks.items(), key=lambda x: x[1])
    
    # Save the table figure if save_path is provided
    if save_path is not None:
        # Construct full path for table
        table_path = os.path.join(save_path, "higher_missing_vars_details_table.png")
        
        # Save the figure
        fig_table.savefig(table_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Detailed table saved to {table_path}")
        
        # Create a CSV table output
        csv_data = []
        for sector in sorted_sector_names:
            if sector_sig_vars[sector]:
                for var_data in sorted(sector_sig_vars[sector], key=lambda x: x['difference'], reverse=True):
                    csv_data.append({
                        'Sector': sector,
                        'Variable': var_data['variable'],
                        'Sector_Missing_Rate': f"{var_data['sector_rate']:.4f}",
                        'Overall_Missing_Rate': f"{var_data['overall_rate']:.4f}",
                        'Difference': f"{var_data['difference']:.4f}",
                        'Chi_square': f"{var_data['chi2']:.4f}",
                        'P_value': f"{var_data['p_value']:.6f}"
                    })
        
        # Convert to DataFrame and save as CSV
        pd.DataFrame(csv_data).to_csv(os.path.join(save_path, "higher_missing_vars_details.csv"), index=False)
        logger.info(f"Detailed CSV exported to {os.path.join(save_path, 'higher_missing_vars_details.csv')}")
        
        # Create and save a sector ranking summary
        plt.figure(figsize=(10, 6))
        sectors, ranks = zip(*sorted_avg_ranks)
        plt.bar(sectors, ranks, color='darkred')
        plt.title('Sectors Ranked by Average Missing Data (Lower Rank = More Missing Data)', fontsize=14)
        plt.xlabel('Sector', fontsize=12)
        plt.ylabel('Average Rank (across all variables)', fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "sector_missing_data_ranking.png"), dpi=dpi, bbox_inches='tight')
    
    # Generate summary of significant missing variables by sector
    sector_summary = []
    for sector in sorted_sector_names:
        if sector_sig_vars[sector]:
            sector_info = {
                'sector': sector,
                'variable_count': sector_sig_counts[sector],
                'variables': []
            }
            for var_data in sorted(sector_sig_vars[sector], key=lambda x: x['difference'], reverse=True):
                sector_info['variables'].append({
                    'name': var_data['variable'],
                    'sector_rate': var_data['sector_rate'],
                    'overall_rate': var_data['overall_rate'],
                    'difference': var_data['difference'],
                    'chi2': var_data['chi2'],
                    'p_value': var_data['p_value']
                })
            sector_summary.append(sector_info)
    
    # Add summary of average ranks
    rank_summary = [{
        'sector': sector, 
        'avg_rank': avg_rank
    } for sector, avg_rank in sorted_avg_ranks]
    
    # Calculate overall stats
    higher_missing_count = sum(count for sector, count in sector_sig_counts.items())
    overall_chi2 = sum(r['chi2'] for r in results.values())
    
    overall_stats = {
        'total_significant_vars': higher_missing_count,
        'combined_chi_square': overall_chi2
    }
    
    logger.info(f"Summary: Found {higher_missing_count} variables with significantly higher missing rates across sectors")
    logger.info(f"Combined Chi-square: {overall_chi2:.2f}")
    
    return overall_chi2, results, {'sector_summary': sector_summary, 'rank_summary': rank_summary, 'overall_stats': overall_stats}


def evaluate_imputation_strategies(df, variables_to_test=None, n_splits=5, random_state=42):
    """
    Evaluate different imputation strategies using k-fold cross-validation
    on artificially created missing data.
    
    Parameters:
    df (pandas.DataFrame): The dataframe with missing values
    variables_to_test (list): List of variables to test imputation on. If None, uses variables with some missing values
    n_splits (int): Number of folds for cross-validation
    random_state (int): Random seed for reproducibility
    
    Returns:
    pandas.DataFrame: DataFrame containing MSE for each imputation method and variable
    """
    # If no variables specified, select numeric variables with some missing values
    if variables_to_test is None:
        variables_to_test = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                          if df[col].isnull().sum() > 0 and df[col].isnull().sum() < len(df)]
    
    # Ensure sector column is available
    if 'gics_sector' not in df.columns:
        raise ValueError("DataFrame must contain 'gics_sector' column for sector-based imputation")
    
    # Initialize results dictionary
    results = {
        'variable': [],
        'global_mean_mse': [],
        'sector_mean_mse': [],
        'sector_median_mse': []
    }
    
    # Create k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Process each variable
    for variable in variables_to_test:
        logger.info(f"Processing variable: {variable}")
            
        # Get complete cases for this variable
        complete_data = df.dropna(subset=[variable])
        
        if len(complete_data) < 100:  # Skip if not enough complete cases
            logger.info(f"  Skipping {variable}: not enough complete cases")
            continue
        
        # Check if the variable has Int64 dtype and convert to float64 for processing
        original_dtype = df[variable].dtype
        is_int64 = str(original_dtype) == 'Int64'
        if is_int64:
            # Create a temporary copy with float64 dtype for processing
            complete_data = complete_data.copy()
            complete_data[variable] = complete_data[variable].astype('float64')
            
        # Initialize MSE lists for this variable
        global_mean_mses = []
        sector_mean_mses = []
        sector_median_mses = []
        
        # Perform k-fold cross-validation
        for train_idx, test_idx in kf.split(complete_data):
            train_data = complete_data.iloc[train_idx]
            test_data = complete_data.iloc[test_idx]
            
            # Create artificial missing data in test set (randomly mask 20% of values)
            np.random.seed(random_state)
            mask = np.random.random(len(test_data)) < 0.2
            test_data_with_missing = test_data.copy()
            true_values = test_data_with_missing.loc[mask, variable].copy()
            test_data_with_missing.loc[mask, variable] = np.nan
            
            # Skip if no values were masked
            if len(true_values) == 0:
                continue
                
            # Method 1: Global Mean Imputation
            global_mean = train_data[variable].mean()
            # Handle Int64 dtype - convert value appropriately
            if is_int64:
                global_mean = int(round(global_mean))
                
            # Create a float copy for imputation
            test_data_with_missing['global_mean_imputed'] = test_data_with_missing[variable].astype('float64').fillna(global_mean)
            
            global_mean_mse = mean_squared_error(
                true_values, 
                test_data_with_missing.loc[mask, 'global_mean_imputed']
            )
            global_mean_mses.append(global_mean_mse)
            
            # Method 2: Sector-specific Mean Imputation
            sector_means = train_data.groupby('gics_sector')[variable].mean()
            # Handle Int64 dtype
            if is_int64:
                sector_means = sector_means.apply(lambda x: int(round(x)))
                
            test_data_with_missing['sector_mean_imputed'] = test_data_with_missing[variable].astype('float64').copy()
            
            for sector in sector_means.index:
                sector_mask = (test_data_with_missing['gics_sector'] == sector) & test_data_with_missing[variable].isnull()
                test_data_with_missing.loc[sector_mask, 'sector_mean_imputed'] = sector_means[sector]
            
            # For any sectors without a mean (if any), use global mean
            test_data_with_missing['sector_mean_imputed'] = test_data_with_missing['sector_mean_imputed'].fillna(global_mean)
            
            sector_mean_mse = mean_squared_error(
                true_values, 
                test_data_with_missing.loc[mask, 'sector_mean_imputed']
            )
            sector_mean_mses.append(sector_mean_mse)
            
            # Method 3: Sector-specific Median Imputation
            sector_medians = train_data.groupby('gics_sector')[variable].median()
            # Handle Int64 dtype
            if is_int64:
                sector_medians = sector_medians.apply(lambda x: int(round(x)))
                
            test_data_with_missing['sector_median_imputed'] = test_data_with_missing[variable].astype('float64').copy()
            
            for sector in sector_medians.index:
                sector_mask = (test_data_with_missing['gics_sector'] == sector) & test_data_with_missing[variable].isnull()
                test_data_with_missing.loc[sector_mask, 'sector_median_imputed'] = sector_medians[sector]
            
            # For any sectors without a median (if any), use global median
            global_median = train_data[variable].median()
            if is_int64:
                global_median = int(round(global_median))
                
            test_data_with_missing['sector_median_imputed'] = test_data_with_missing['sector_median_imputed'].fillna(global_median)
            
            sector_median_mse = mean_squared_error(
                true_values, 
                test_data_with_missing.loc[mask, 'sector_median_imputed']
            )
            sector_median_mses.append(sector_median_mse)
        
        # Calculate average MSE across folds
        if global_mean_mses:  # Only add results if we have valid MSEs
            results['variable'].append(variable)
            results['global_mean_mse'].append(np.mean(global_mean_mses))
            results['sector_mean_mse'].append(np.mean(sector_mean_mses))
            results['sector_median_mse'].append(np.mean(sector_median_mses))
            
            logger.info(f"  Global Mean MSE: {np.mean(global_mean_mses):.2f}")
            logger.info(f"  Sector Mean MSE: {np.mean(sector_mean_mses):.2f}")
            logger.info(f"  Sector Median MSE: {np.mean(sector_median_mses):.2f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def plot_imputation_comparison(results_df, save_path=None, filename="imputation_comparison.png", dpi=300):
    """
    Generate visualizations comparing imputation methods.
    
    Parameters:
    results_df (pandas.DataFrame): DataFrame with imputation results
    save_path (str, optional): Directory path to save the figure
    filename (str, optional): Filename for the saved figure
    dpi (int, optional): Resolution for the saved figure
    """
    # Melt the dataframe for easier plotting
    melted_df = pd.melt(
        results_df, 
        id_vars=['variable'], 
        value_vars=['global_mean_mse', 'sector_mean_mse', 'sector_median_mse'],
        var_name='method', value_name='mse'
    )
    
    # Clean up method names for display
    method_map = {
        'global_mean_mse': 'Global Mean',
        'sector_mean_mse': 'Sector Mean',
        'sector_median_mse': 'Sector Median'
    }
    melted_df['method'] = melted_df['method'].map(method_map)
    
    # Create a table visualization with dynamic size based on number of variables
    # Use a maximum height and scroll if more variables
    max_rows_visible = 25
    table_height = min(len(results_df) * 0.5 + 2, max_rows_visible)
    plt.figure(figsize=(14, table_height))
    
    table_data = results_df.copy()
    # Format large numbers using scientific notation
    for col in table_data.columns[1:]:  # Skip the variable name column
        table_data[col] = table_data[col].apply(lambda x: f"{x:.2e}" if abs(x) > 1000 else round(x, 2))
    
    # Highlight the best method (lowest MSE) for each variable
    best_method = results_df.iloc[:, 1:].idxmin(axis=1)
    
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    table = plt.table(
        cellText=table_data.values,
        colLabels=['Variable', 'Global Mean', 'Sector Mean', 'Sector Median'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)
    
    # Highlight best values
    for i, method in enumerate(best_method):
        col_idx = table_data.columns.get_loc(method)
        cell = table[(i+1, col_idx)]  # +1 to account for header row
        cell.set_facecolor('#AED6F1')  # Light blue highlight
    
    plt.title('Imputation Method Comparison: Mean Squared Error', fontsize=14)
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Construct full path
        full_path = os.path.join(save_path, filename)
        
        # Save the figure
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Imputation comparison figure saved to {full_path}")
        
        # Save the data as CSV
        csv_path = os.path.join(save_path, 'imputation_results.csv')
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Imputation results saved to {csv_path}")
    
    fig = plt.gcf()  # Get current figure
    plt.close()
    return fig


def apply_best_imputation(df, variables_to_impute=None, method='sector_mean'):
    """
    Apply the best imputation method to the dataset.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with missing values
    variables_to_impute (list): List of variables to impute. If None, imputes all variables with missing values
    method (str): Imputation method to use ('global_mean', 'sector_mean', or 'sector_median')
    
    Returns:
    pandas.DataFrame: DataFrame with imputed values
    """
    # Create a copy of the dataframe
    imputed_df = df.copy()
    
    # If no variables specified, select numeric variables with some missing values
    if variables_to_impute is None:
        variables_to_impute = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                             if df[col].isnull().sum() > 0]
    
    logger.info(f"Imputing {len(variables_to_impute)} variables using {method} method")
    
    # Ensure sector column is available if using sector-based methods
    if method.startswith('sector') and 'gics_sector' not in df.columns:
        raise ValueError("DataFrame must contain 'gics_sector' column for sector-based imputation")
    
    # Apply imputation for each variable
    for variable in variables_to_impute:
        # Check if the variable has Int64 dtype
        original_dtype = df[variable].dtype
        is_int64 = str(original_dtype) == 'Int64'
        
        if method == 'global_mean':
            # Global mean imputation
            global_mean = df[variable].mean()
            
            # Convert to integer if the target is Int64
            if is_int64:
                global_mean = int(round(global_mean))
                
            # Convert to float first, then fill, then convert back if needed
            if is_int64:
                temp_series = imputed_df[variable].astype('float64').fillna(global_mean)
                imputed_df[variable] = temp_series.astype('Int64')
            else:
                imputed_df[variable] = imputed_df[variable].fillna(global_mean)
            
        elif method == 'sector_mean':
            # Sector-specific mean imputation
            sector_means = df.groupby('gics_sector')[variable].mean()
            
            # Convert to integer if the target is Int64
            if is_int64:
                sector_means = sector_means.apply(lambda x: int(round(x)))
                # Create a float copy for imputation
                temp_var = imputed_df[variable].astype('float64')
                
                for sector in sector_means.index:
                    sector_mask = (imputed_df['gics_sector'] == sector) & imputed_df[variable].isnull()
                    temp_var.loc[sector_mask] = sector_means[sector]
                
                # For any remaining NaN values, use global mean
                if temp_var.isnull().any():
                    global_mean = int(round(df[variable].mean()))
                    temp_var = temp_var.fillna(global_mean)
                
                # Convert back to Int64
                imputed_df[variable] = temp_var.astype('Int64')
            else:
                for sector in sector_means.index:
                    sector_mask = (imputed_df['gics_sector'] == sector) & imputed_df[variable].isnull()
                    imputed_df.loc[sector_mask, variable] = sector_means[sector]
                
                # For any remaining NaN values, use global mean
                if imputed_df[variable].isnull().any():
                    global_mean = df[variable].mean()
                    imputed_df[variable] = imputed_df[variable].fillna(global_mean)
                
        elif method == 'sector_median':
            # Sector-specific median imputation
            sector_medians = df.groupby('gics_sector')[variable].median()
            
            # Convert to integer if the target is Int64
            if is_int64:
                sector_medians = sector_medians.apply(lambda x: int(round(x)))
                # Create a float copy for imputation
                temp_var = imputed_df[variable].astype('float64')
                
                for sector in sector_medians.index:
                    sector_mask = (imputed_df['gics_sector'] == sector) & imputed_df[variable].isnull()
                    temp_var.loc[sector_mask] = sector_medians[sector]
                
                # For any remaining NaN values, use global median
                if temp_var.isnull().any():
                    global_median = int(round(df[variable].median()))
                    temp_var = temp_var.fillna(global_median)
                
                # Convert back to Int64
                imputed_df[variable] = temp_var.astype('Int64')
            else:
                for sector in sector_medians.index:
                    sector_mask = (imputed_df['gics_sector'] == sector) & imputed_df[variable].isnull()
                    imputed_df.loc[sector_mask, variable] = sector_medians[sector]
                
                # For any remaining NaN values, use global median
                if imputed_df[variable].isnull().any():
                    global_median = df[variable].median()
                    imputed_df[variable] = imputed_df[variable].fillna(global_median)
        else:
            raise ValueError("Method must be one of: 'global_mean', 'sector_mean', 'sector_median'")
    
    # Verify imputation worked
    missing_after = imputed_df[variables_to_impute].isnull().sum().sum()
    logger.info(f"Missing values after imputation: {missing_after}")
    
    return imputed_df


def prepare_data_for_analysis(df):
    """
    Prepare data for analysis by inferring appropriate data types.
    
    Parameters:
    df (pandas.DataFrame): Raw DataFrame
    
    Returns:
    tuple: (df, categorical_cols, numerical_cols)
    """
    # Automatically infer best dtypes for each column
    df = df.convert_dtypes()
    
    # Define categorical and numerical variables
    categorical_cols = [var for var in df.columns if df[var].dtype=='string']
    numerical_cols = [var for var in df.columns if df[var].dtype!='string']
    
    logger.info(f"Identified {len(categorical_cols)} categorical variables and {len(numerical_cols)} numerical variables")
    
    return df, categorical_cols, numerical_cols


def save_processed_data(df, categorical_cols, numerical_cols, output_dir):
    """
    Save processed data to output directory.
    
    Parameters:
    df (pandas.DataFrame): Processed DataFrame
    categorical_cols (list): List of categorical column names
    numerical_cols (list): List of numerical column names
    output_dir (str): Output directory path
    
    Returns:
    tuple: Paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save categorical data
    df_cat = df[categorical_cols]
    if 'issuer_name' in df.columns:
        df_cat = df_cat.set_index(df['issuer_name'])
    
    cat_path = os.path.join(output_dir, 'categorical.csv')
    df_cat.to_csv(cat_path, index=True)
    logger.info(f"Categorical data saved to {cat_path}")
    
    # Save numerical data
    df_num = df[numerical_cols]
    if 'issuer_name' in df.columns:
        df_num = df_num.set_index(df['issuer_name'])
    
    num_path = os.path.join(output_dir, 'numerical.csv')
    df_num.to_csv(num_path, index=True)
    logger.info(f"Numerical data saved to {num_path}")
    
    # Save combined data
    combined_path = os.path.join(output_dir, 'imputed_data.csv')
    df.to_csv(combined_path, index=False)
    logger.info(f"Complete imputed data saved to {combined_path}")
    
    return cat_path, num_path, combined_path