"""Pipeline orchestrator for ESG EDA.

This module provides the main pipeline class that coordinates all analysis steps.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
from pathlib import Path
import logging
import time
from datetime import datetime

from ..analysis.missing_values import MissingValuesAnalyzer
from ..analysis.outlier_detection import OutlierDetector
from ..analysis.feature_engineering import FeatureEngineer
from ..core.config import get_settings
from ..utils.logging import setup_logging, get_logger

logger = logging.getLogger(__name__)


class ESGPipeline:
    """Main pipeline orchestrator for ESG EDA.
    
    This class coordinates all analysis steps and manages the data flow
    through the pipeline stages.
    """
    
    def __init__(self, 
                 input_path: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None):
        """Initialize the pipeline.
        
        Args:
            input_path: Path to input data file
            output_dir: Base directory for outputs
        """
        self.settings = get_settings()
        self.input_path = Path(input_path) if input_path else self.settings.get_input_path()
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        
        # Pipeline state
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.execution_times: Dict[str, float] = {}
        self.results: Dict[str, Any] = {}
        
        # Setup logging
        setup_logging(self.settings.logging)
        
    def load_data(self) -> pd.DataFrame:
        """Load input data.
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.input_path}")
        start_time = time.time()
        
        self.data = pd.read_csv(self.input_path)
        
        self.execution_times['data_loading'] = time.time() - start_time
        logger.info(f"Loaded {len(self.data)} rows with {len(self.data.columns)} columns")
        
        return self.data
    
    def run_missing_values_analysis(self) -> Dict[str, Any]:
        """Run missing values analysis step.
        
        Returns:
            Analysis results
        """
        logger.info("Running missing values analysis")
        start_time = time.time()
        
        analyzer = MissingValuesAnalyzer(self.data)
        results = analyzer.analyze()
        
        # Generate visualizations
        viz_dir = self.output_dir / "visualizations" / "missing_values"
        analyzer.visualize(viz_dir)
        
        # Apply imputation
        self.processed_data = analyzer.impute()
        
        # Save results
        output_dir = self.output_dir / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data.to_csv(output_dir / "imputed_data.csv", index=False)
        
        self.execution_times['missing_values'] = time.time() - start_time
        self.results['missing_values'] = results
        
        return results
    
    def run_outlier_detection(self) -> Dict[str, Any]:
        """Run outlier detection step.
        
        Returns:
            Detection results
        """
        logger.info("Running outlier detection")
        start_time = time.time()
        
        # Use processed data if available, otherwise original
        data = self.processed_data if self.processed_data is not None else self.data
        
        detector = OutlierDetector(data)
        results = detector.detect_all_methods(
            iqr_multiplier=self.settings.pipeline.iqr_multiplier,
            z_threshold=self.settings.pipeline.z_score_threshold
        )
        
        # Generate visualizations
        viz_dir = self.output_dir / "visualizations" / "outliers"
        detector.visualize(viz_dir)
        
        # Save results
        detector.save_results(self.output_dir / "data" / "processed")
        
        self.execution_times['outlier_detection'] = time.time() - start_time
        self.results['outliers'] = results
        
        return results
    
    def run_feature_engineering(self) -> Dict[str, Any]:
        """Run feature engineering step.
        
        Returns:
            Engineering results
        """
        logger.info("Running feature engineering")
        start_time = time.time()
        
        # Use processed data if available
        data = self.processed_data if self.processed_data is not None else self.data
        
        engineer = FeatureEngineer(data)
        
        # Identify features
        feature_types = engineer.identify_features()
        
        # Apply encoding
        engineer.encode_categorical(
            max_categories=self.settings.pipeline.one_hot_max_categories
        )
        
        # Apply transformation
        engineer.transform_numerical(
            threshold=self.settings.pipeline.yeo_johnson_threshold
        )
        
        # Combine all features
        self.processed_data = engineer.combine_all_features()
        
        # Generate visualizations
        viz_dir = self.output_dir / "visualizations" / "features"
        engineer.visualize_feature_expansion(viz_dir)
        
        # Save results
        output_dir = self.output_dir / "data" / "processed"
        self.processed_data.to_csv(output_dir / "engineered_features.csv", index=False)
        engineer.save_feature_info(output_dir / "feature_info.json")
        
        self.execution_times['feature_engineering'] = time.time() - start_time
        self.results['features'] = {
            'feature_types': feature_types,
            'feature_info': engineer.feature_info
        }
        
        return self.results['features']
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline.
        
        Returns:
            All pipeline results
        """
        logger.info("Starting full ESG EDA pipeline")
        pipeline_start = time.time()
        
        # Load data
        self.load_data()
        
        # Run each step
        self.run_missing_values_analysis()
        self.run_outlier_detection()
        self.run_feature_engineering()
        
        # Save final processed data
        output_path = self.output_dir / "data" / "combined_df_for_ml_models.csv"
        self.processed_data.to_csv(output_path, index=False)
        
        # Calculate total time
        self.execution_times['total'] = time.time() - pipeline_start
        
        # Save execution report
        self._save_execution_report()
        
        logger.info(f"Pipeline completed in {self.execution_times['total']:.2f} seconds")
        
        return self.results
    
    def _save_execution_report(self) -> None:
        """Save pipeline execution report."""
        report = {
            'execution_date': datetime.now().isoformat(),
            'input_file': str(self.input_path),
            'output_directory': str(self.output_dir),
            'execution_times': self.execution_times,
            'data_shape': {
                'original': self.data.shape if self.data is not None else None,
                'processed': self.processed_data.shape if self.processed_data is not None else None
            }
        }
        
        # Save as JSON
        import json
        report_path = self.output_dir / "pipeline_execution_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Execution report saved to {report_path}")