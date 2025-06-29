"""
Generate a sample dataset with missing values for testing the missing_values.py module.
"""
import pandas as pd
import numpy as np
import os

# Create a sample dataset
np.random.seed(42)

# Define the number of samples
n_samples = 200

# Create main dataframe
df = pd.DataFrame()

# Company ID
df['company_id'] = range(1, n_samples + 1)

# Year
df['year'] = np.random.choice([2018, 2019, 2020, 2021, 2022], size=n_samples)

# Sector - use 'sector' to match the script and 'gics_sector' to match the data
df['gics_sector'] = np.random.choice(['Technology', 'Healthcare', 'Financial Services', 
                                      'Consumer Goods', 'Energy', 'Utilities'], 
                                     size=n_samples)

# Add numeric variables with missing values
sectors = df['gics_sector'].unique()
for col_name in ['esg_score', 'carbon_emissions', 'revenue_growth', 'profit_margin', 'employee_turnover']:
    # Add sector-specific missing patterns
    base_values = np.random.normal(0, 1, n_samples)
    
    # Scale values by sector to create patterns
    for sector in sectors:
        sector_idx = df['gics_sector'] == sector
        if col_name == 'esg_score':
            scale = np.random.uniform(0.8, 1.2)
            base_values[sector_idx] = np.random.normal(5, 1, sum(sector_idx)) * scale
            # Add missing values with varying probability by sector
            if sector == 'Energy':  # Higher missing rate for Energy sector
                missing_mask = np.random.random(sum(sector_idx)) < 0.3
                base_values[sector_idx] = np.where(missing_mask, np.nan, base_values[sector_idx])
            elif sector == 'Technology':  # Lower missing rate for Technology
                missing_mask = np.random.random(sum(sector_idx)) < 0.05
                base_values[sector_idx] = np.where(missing_mask, np.nan, base_values[sector_idx])
            else:  # Medium missing rate for other sectors
                missing_mask = np.random.random(sum(sector_idx)) < 0.15
                base_values[sector_idx] = np.where(missing_mask, np.nan, base_values[sector_idx])
        elif col_name == 'carbon_emissions':
            scale = np.random.uniform(0.5, 1.5)
            base_values[sector_idx] = np.random.normal(50, 20, sum(sector_idx)) * scale
            # Add missing values with varying probability by sector
            if sector == 'Utilities':  # Higher missing rate for Utilities
                missing_mask = np.random.random(sum(sector_idx)) < 0.35
                base_values[sector_idx] = np.where(missing_mask, np.nan, base_values[sector_idx])
            else:  # Lower missing rate for other sectors
                missing_mask = np.random.random(sum(sector_idx)) < 0.1
                base_values[sector_idx] = np.where(missing_mask, np.nan, base_values[sector_idx])
        else:
            # Random scaling for other columns
            scale = np.random.uniform(0.7, 1.3)
            base_values[sector_idx] = base_values[sector_idx] * scale
            # Add some random missing values
            missing_mask = np.random.random(n_samples) < 0.15
            base_values = np.where(missing_mask, np.nan, base_values)
    
    df[col_name] = base_values

# Add categorical variables
cat_values = {
    'data_quality': ['High', 'Medium', 'Low', None],
    'reporting_framework': ['GRI', 'SASB', 'TCFD', 'Other', None],
    'verification_status': ['Verified', 'Unverified', None]
}

for col, values in cat_values.items():
    weights = [0.4, 0.3, 0.2, 0.1] if None in values else [1/len(values)] * len(values)
    df[col] = np.random.choice(values, size=n_samples, p=weights)

# Save to CSV
output_path = os.path.join('data', 'raw', 'df_cleaned_test.csv')
df.to_csv(output_path, index=False)

print(f"Sample dataset created and saved to {output_path}")
print(f"Shape: {df.shape}")
print(f"Missing values by column:")
print(df.isnull().sum())