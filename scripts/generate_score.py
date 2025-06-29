#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate ESG Score CSV File

This script generates a score.csv file containing the ORIGINAL ESG scores (0-10 scale)
from the raw data file. The score.csv file contains the issuer names and their 
corresponding ESG scores, which can be used as the target variable for model training.
"""

import os
import sys
import pandas as pd
import numpy as np

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location import Location

def generate_score_csv(location):
    """
    Generate a score.csv file containing ORIGINAL ESG scores from raw data.
    
    Parameters
    ----------
    location : Location
        Location object for handling file paths
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print("Generating score.csv file with original ESG scores (0-10 scale)...")
    
    # Load the raw data to get original ESG scores
    raw_data_path = location.get_path('data/raw', 'df_cleaned.csv')
    
    try:
        df = pd.read_csv(raw_data_path)
        print(f"Loaded raw data from {raw_data_path}")
        
        # Set issuer_name as index to match other files
        df.set_index('issuer_name', inplace=True)
        
        # Extract only the esg_score column
        if 'esg_score' in df.columns:
            score_df = df[['esg_score']].copy()
            
            # Verify the scores are in the expected range (0-10)
            print(f"ESG score statistics:")
            print(f"  Range: {score_df['esg_score'].min():.2f} to {score_df['esg_score'].max():.2f}")
            print(f"  Mean: {score_df['esg_score'].mean():.2f}")
            print(f"  Std Dev: {score_df['esg_score'].std():.2f}")
            print(f"  Number of companies: {len(score_df)}")
            
            # Check if any values are outside 0-10 range
            if score_df['esg_score'].min() < 0 or score_df['esg_score'].max() > 10:
                print("WARNING: Some ESG scores are outside the expected 0-10 range!")
        else:
            print("Error: 'esg_score' column not found in raw data.")
            return False
            
    except FileNotFoundError:
        print(f"Error: Could not find raw data file at {raw_data_path}")
        return False
    except Exception as e:
        print(f"Error loading raw data: {e}")
        return False
    
    # Save the score data to both locations for compatibility
    output_paths = [
        location.get_path('data', 'score.csv'),
        location.get_path('data/ml_output/raw', 'score.csv')
    ]
    
    for output_path in output_paths:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            score_df.to_csv(output_path, index=True)
            print(f"Score data saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save to {output_path}: {e}")
    
    return True

if __name__ == "__main__":
    # Initialize location
    base_dir = os.getcwd()    
    location = Location(base_dir)
    
    # Generate score.csv
    success = generate_score_csv(location)
    
    if success:
        print("Score file generation completed successfully.")
    else:
        print("Score file generation failed.")
        sys.exit(1)