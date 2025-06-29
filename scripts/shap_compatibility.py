#!/usr/bin/env python3
"""
SHAP Compatibility Helper Script

This script demonstrates how to fix NumPy 2.0 compatibility issues with SHAP.
It provides a simple helper function to patch NumPy for SHAP usage.
"""

import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_numpy_for_shap():
    """
    Patch NumPy for SHAP compatibility with NumPy 2.0+
    
    This function adds a compatibility layer for the removed obj2sctype function
    in NumPy 2.0 which is still used by some versions of SHAP.
    
    Returns:
    --------
    bool
        True if patching was applied, False if not needed
    """
    # Check if we're using NumPy 2.0+
    if hasattr(np, 'version') and int(np.version.version.split('.')[0]) >= 2:
        # Check if obj2sctype is missing
        if not hasattr(np, 'obj2sctype'):
            # Create a compatible version using np.dtype
            def obj2sctype_compat(obj):
                try:
                    return np.dtype(obj).type
                except (TypeError, ValueError):
                    return None
                
            # Add the function to NumPy namespace
            np.obj2sctype = obj2sctype_compat
            logger.info("Added NumPy 2.0 compatibility patch for SHAP")
            return True
        else:
            logger.info("NumPy obj2sctype already exists, no patching needed")
            return False
    else:
        logger.info(f"NumPy version {np.version.version} doesn't need patching")
        return False

def shap_example():
    """
    Simple example showing how to use SHAP with the compatibility patch
    """
    # Apply the patch first
    patch_numpy_for_shap()
    
    # Now import and use SHAP
    try:
        import shap
        from sklearn.ensemble import IsolationForest
        import pandas as pd
        
        # Create a simple dataset
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        
        # Train an Isolation Forest model
        model = IsolationForest(random_state=42)
        model.fit(X)
        
        # Create a SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Create a summary plot
        shap.summary_plot(shap_values, X, show=False)
        
        # Save the plot
        import matplotlib.pyplot as plt
        plt.savefig('shap_summary.png')
        plt.close()
        
        logger.info("Successfully created SHAP visualization with NumPy compatibility patch")
        return True
    except Exception as e:
        logger.error(f"Error using SHAP: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the example
    print(f"NumPy version: {np.version.version}")
    
    # Patch NumPy for SHAP
    patched = patch_numpy_for_shap()
    print(f"NumPy patched: {patched}")
    
    # Run SHAP example if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--run-example":
        success = shap_example()
        print(f"SHAP example run successfully: {success}")
    else:
        print("Use --run-example flag to run a SHAP visualization test")