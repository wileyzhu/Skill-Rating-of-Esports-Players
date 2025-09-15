"""
Main Pipeline for League of Legends Player Skill Rating Analysis
Orchestrates the complete machine learning pipeline from data collection to analysis
"""

import os
import sys
import argparse
import logging
from typing import Optional, List
import pandas as pd
import numpy as np

# Import our modules
from data_collection import MatchScraper
from data_preprocessing import DataProcessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from shap_analysis import SHAPAnalyzer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('lol_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


class LoLAnalysisPipeline:
    """Complete analysis pipeline for League of Legends player skill rating"""
    
    def __init__(self, base_path: str = "/users/wiley/Documents/Downloads",
                 random_state: int = 42):
        self.base_path = base_path
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.scraper = None
        self.processor = DataProcessor()
        self.trainer = ModelTrainer(random_state=random_state)
        self.evaluator = ModelEvaluator()
        self.shap_analyzer = SHAPAnalyzer()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.role_datasets = None
        self.trainer_results = None
        
    def run_data_collection(self, game_id_ranges: List[tuple], 
                           use_existing: bool = True) -> pd.DataFrame:
        """Run data collection phase"""
        self.logger.info("Starting data collection phase...")
        
        final_data_path = f"{self.base_path}/all_matches_final.csv"
        
        if use_existing and os.path.exists(final_data_path):
            self.logger.info(f"Loading existing data from {final_data_path}")
            self.raw_data = pd.read_csv(final_data_path)
            return self.raw_data
        
        # Initialize scraper
        self.scraper = MatchScraper(headless=True)
        
        # Collect data for each range
        all_data = []
        for i, (start_id, end_id) in enumerate(game_id_ranges):
            self.logger.info(f"Scraping game IDs {start_id} to {end_id} (batch {i})")
            game_ids = np.arange(start_id, end_id)
            batch_data = self.scraper.scrape_batch(game_ids, i, self.base_path)
            if not batch_data.empty:
                all_data.append(batch_data)
        
        # Merge all batches
        if all_data:
            self.raw_data = pd.concat(all_data, ignore_index=True)
            self.raw_data.to_csv(final_data_path, index=False)
            self.logger.info(f"Collected {len(self.raw_data)} total records")
        else:
            self.logger.error("No data collected!")
            raise ValueError("Data collection failed")
        
        return self.raw_data
    
    def run_data_preprocessing(self, use_existing_data: bool = True) -> tuple:
        """Run data preprocessing phase"""
        self.logger.info("Starting data preprocessing phase...")
        
        if use_existing_data and self.raw_data is None:
            # Try to load existing processed data
            try:
                self.processed_data, self.role_datasets = self.processor.process_full_pipeline(self.base_path)
            except Exception as e:
                self.logger.error(f"Failed to load existing data: {e}")
                raise
        else:
            # Process raw data
            if self.raw_data is None:
                raise ValueError("No raw data available for preprocessing")
            
            # Run preprocessing pipeline
            self.processed_data = self.processor.assign_bo5_series(self.raw_data)
            self.processed_data = self.processor.engineer_features(self.processed_data)
            self.role_datasets = self.processor.prepare_role_datasets(self.processed_data)
        
        # Log preprocessing results
        self.logger.info(f"Processed data shape: {self.processed_data.shape}")
        for role, data in self.role_datasets.items():
            win_rate = data['Win'].mean() if 'Win' in data.columns else 0
            self.logger.info(f"{role}: {len(data)} samples, win rate: {win_rate:.3f}")
        
        return self.processed_data, self.role_datasets
    
    def run_model_training(self, min_samples: int = 100) -> dict:
        """Run model training phase"""
        self.logger.info("Starting model training phase...")
        
        if self.role_datasets is None:
            raise ValueError("No role datasets available for training")
        
        feature_names = self.processor.get_feature_names()
        self.trainer_results = {}
        
        # Train models for each role
        for role, role_data in self.role_datasets.items():
            if len(role_data) < min_samples:
                self.logger.warning(f"Skipping {role} due to insufficient data ({len(role_data)} samples)")
                continue
            
            self.logger.info(f"Training models for {role} role...")
            
            # Prepare features and target
            X = role_data[feature_names].copy()
            y = role_data['Win'].copy()
            
            # Remove any remaining NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < min_samples:
                self.logger.warning(f"Insufficient clean data for {role} after preprocessing")
                continue
            
            # Train models
            try:
                role_results = self.trainer.train_role_models(X, y, role)
                self.trainer_results[role] = role_results
                self.logger.info(f"Successfully trained models for {role}")
            except Exception as e:
                self.logger.error(f"Failed to train models for {role}: {e}")
                continue
        
        # Save models and results
        self.trainer.save_models()
        self.trainer.save_results()
        
        # Display best models
        best_models = self.trainer.get_best_model_per_role()
        self.logger.info("Best models per role:")
        for role, (model_name, model) in best_models.items():
            if role in self.trainer.results:
                auc = self.trainer.results[role][model_name]['auc_roc']
                self.logger.info(f"{role}: {model_name} (AUC: {auc:.3f})")
        
        return self.trainer_results
    
    def run_model_evaluation(self) -> dict:
        """Run model evaluation phase"""
        self.logger.info("Starting model evaluation phase...")
        
        if self.trainer_results is None:
            raise ValueError("No trained models available for evaluation")
        
        # Run comprehensive evaluation
        evaluation_results = self.evaluator.run_comprehensive_evaluation(self.trainer_results)
        
        self.logger.info("Model evaluation completed successfully")
        return evaluation_results
    
    def run_shap_analysis(self) -> None:
        """Run SHAP interpretability analysis"""
        self.logger.info("Starting SHAP analysis phase...")
        
        if self.trainer_results is None:
            raise ValueError("No trained models available for SHAP analysis")
        
        feature_names = self.processor.get_feature_names()
        
        # Run complete SHAP analysis
        self.shap_analyzer.run_complete_shap_analysis(self.trainer_results, feature_names)
        
        self.logger.info("SHAP analysis completed successfully")
    
    def run_complete_pipeline(self, game_id_ranges: Optional[List[tuple]] = None,
                            skip_data_collection: bool = True,
                            skip_training: bool = False,
                            skip_evaluation: bool = False,
                            skip_shap: bool = False) -> dict:
        """Run the complete analysis pipeline"""
        self.logger.info("Starting complete LoL analysis pipeline...")
        
        results = {}
        
        try:
            # Phase 1: Data Collection (optional)
            if not skip_data_collection and game_id_ranges:
                self.run_data_collection(game_id_ranges, use_existing=True)
                results['data_collection'] = True
            
            # Phase 2: Data Preprocessing
            self.run_data_preprocessing(use_existing_data=skip_data_collection)
            results['preprocessing'] = True
            
            # Phase 3: Model Training
            if not skip_training:
                self.run_model_training()
                results['training'] = True
            
            # Phase 4: Model Evaluation
            if not skip_evaluation and not skip_training:
                evaluation_results = self.run_model_evaluation()
                results['evaluation'] = evaluation_results
            
            # Phase 5: SHAP Analysis
            if not skip_shap and not skip_training:
                self.run_shap_analysis()
                results['shap_analysis'] = True
            
            self.logger.info("Complete pipeline executed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        return results
    
    def generate_final_report(self) -> str:
        """Generate a final analysis report"""
        report = []
        report.append("# League of Legends Player Skill Rating Analysis Report")
        report.append("=" * 60)
        
        if self.processed_data is not None:
            report.append(f"\n## Dataset Summary")
            report.append(f"Total matches analyzed: {len(self.processed_data)}")
            report.append(f"Unique games: {self.processed_data['Game_ID'].nunique()}")
            
            if 'bo5' in self.processed_data.columns:
                bo5_games = self.processed_data['bo5'].sum()
                report.append(f"Bo5 games identified: {bo5_games}")
        
        if self.role_datasets is not None:
            report.append(f"\n## Role Distribution")
            for role, data in self.role_datasets.items():
                win_rate = data['Win'].mean() if 'Win' in data.columns else 0
                report.append(f"{role}: {len(data)} players, {win_rate:.1%} win rate")
        
        if hasattr(self.trainer, 'results') and self.trainer.results:
            report.append(f"\n## Model Performance Summary")
            best_models = self.trainer.get_best_model_per_role()
            for role, (model_name, model) in best_models.items():
                if role in self.trainer.results:
                    metrics = self.trainer.results[role][model_name]
                    report.append(f"{role}: {model_name}")
                    report.append(f"  - Accuracy: {metrics['accuracy']:.3f}")
                    report.append(f"  - F1-Score: {metrics['f1_score']:.3f}")
                    report.append(f"  - AUC-ROC: {metrics['auc_roc']:.3f}")
        
        report.append(f"\n## Files Generated")
        report.append("- Trained models saved to models/")
        report.append("- Evaluation results saved to results/")
        report.append("- SHAP analysis plots saved to shap_plots/")
        
        report_text = "\n".join(report)
        
        # Save report
        with open("analysis_report.txt", "w") as f:
            f.write(report_text)
        
        return report_text


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description="LoL Player Skill Rating Analysis Pipeline")
    parser.add_argument("--skip-collection", action="store_true", 
                       help="Skip data collection phase")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training phase")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip model evaluation phase")
    parser.add_argument("--skip-shap", action="store_true",
                       help="Skip SHAP analysis phase")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level")
    parser.add_argument("--base-path", default="/users/wiley/Documents/Downloads",
                       help="Base path for data files")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Initialize pipeline
    pipeline = LoLAnalysisPipeline(
        base_path=args.base_path,
        random_state=args.random_state
    )
    
    # Define game ID ranges for data collection (if needed)
    game_id_ranges = [
        (53697, 54000),
        (54000, 55000),
        (55000, 56000),
        # Add more ranges as needed
    ]
    
    try:
        # Run pipeline
        results = pipeline.run_complete_pipeline(
            game_id_ranges=game_id_ranges if not args.skip_collection else None,
            skip_data_collection=args.skip_collection,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            skip_shap=args.skip_shap
        )
        
        # Generate final report
        report = pipeline.generate_final_report()
        print("\n" + report)
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()