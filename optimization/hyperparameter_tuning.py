"""
Hyperparameter tuning for VinBigData Chest X-ray models
"""
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import time

from models.classification_model import ChestXrayClassifier, MultiLabelLoss, ClassificationMetrics
from training.trainer import ClassificationTrainer
from data_preparation.dataset import create_classification_datasets
from config import MODEL_CONFIG, TRAINING_CONFIG

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Hyperparameter tuning using Optuna
    """
    
    def __init__(self, 
                 train_dataset,
                 val_dataset,
                 class_names: List[str],
                 device: str = "cuda",
                 n_trials: int = 50,
                 timeout: Optional[int] = None):
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.class_names = class_names
        self.device = device
        self.n_trials = n_trials
        self.timeout = timeout
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',  # Maximize validation F1 score
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization"""
        
        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial)
        
        try:
            # Create model with suggested parameters
            model = ChestXrayClassifier(
                num_classes=len(self.class_names),
                backbone=params['backbone'],
                pretrained=True,
                dropout_rate=params['dropout_rate']
            )
            
            # Create data loaders
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Create trainer
            trainer = ClassificationTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                use_wandb=False
            )
            
            # Update optimizer with suggested parameters
            trainer.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            
            # Train for a few epochs
            best_val_f1 = 0.0
            for epoch in range(params['epochs']):
                # Train
                trainer.train_epoch()
                
                # Validate
                val_metrics = trainer.validate_epoch()
                val_f1 = val_metrics.get('macro_f1', 0.0)
                
                # Update best score
                best_val_f1 = max(best_val_f1, val_f1)
                
                # Report intermediate result
                trial.report(val_f1, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_val_f1
            
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial"""
        
        params = {
            # Model architecture
            'backbone': trial.suggest_categorical(
                'backbone', 
                ['efficientnet-b0', 'efficientnet-b2', 'efficientnet-b4', 
                 'resnet50', 'resnet101']
            ),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            
            # Training parameters
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'epochs': trial.suggest_int('epochs', 5, 20),
            
            # Loss function parameters
            'focal_alpha': trial.suggest_float('focal_alpha', 0.1, 0.5),
            'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),
            
            # Data augmentation
            'augmentation_strength': trial.suggest_categorical(
                'augmentation_strength', ['light', 'medium', 'heavy']
            )
        }
        
        return params
    
    def optimize(self, save_dir: str = "hyperparameter_tuning") -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation F1: {best_value:.4f}")
        
        # Save results
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(self.study.trials),
            'study_summary': {
                'best_trial': self.study.best_trial.number,
                'best_value': best_value,
                'best_params': best_params
            }
        }
        
        # Save to file
        results_path = save_dir / "optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save study
        study_path = save_dir / "study.pkl"
        optuna.study.save_study(self.study, study_path)
        
        # Create visualization plots
        self.create_optimization_plots(save_dir)
        
        return results
    
    def create_optimization_plots(self, save_dir: Path):
        """Create optimization visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Optimization history
            fig, ax = plt.subplots(figsize=(10, 6))
            trial_numbers = [trial.number for trial in self.study.trials]
            trial_values = [trial.value for trial in self.study.trials if trial.value is not None]
            
            ax.plot(trial_numbers[:len(trial_values)], trial_values, 'o-', alpha=0.7)
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Validation F1 Score')
            ax.set_title('Hyperparameter Optimization History')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / "optimization_history.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Parameter importance
            try:
                importance = optuna.importance.get_param_importances(self.study)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                params = list(importance.keys())
                values = list(importance.values())
                
                ax.barh(params, values)
                ax.set_xlabel('Importance')
                ax.set_title('Parameter Importance')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(save_dir / "parameter_importance.png", dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logger.warning(f"Could not create parameter importance plot: {str(e)}")
            
            logger.info(f"Optimization plots saved to {save_dir}")
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")

class GridSearchTuner:
    """
    Grid search hyperparameter tuning (alternative to Optuna)
    """
    
    def __init__(self, 
                 train_dataset,
                 val_dataset,
                 class_names: List[str],
                 device: str = "cuda"):
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.class_names = class_names
        self.device = device
    
    def grid_search(self, 
                   param_grid: Dict[str, List[Any]],
                   epochs: int = 10,
                   save_dir: str = "grid_search") -> Dict[str, Any]:
        """Perform grid search optimization"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all parameter combinations
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        logger.info(f"Grid search with {len(param_combinations)} combinations...")
        
        results = []
        best_score = 0.0
        best_params = None
        
        for i, param_combination in enumerate(param_combinations):
            params = dict(zip(param_names, param_combination))
            
            logger.info(f"Trial {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Create model
                model = ChestXrayClassifier(
                    num_classes=len(self.class_names),
                    backbone=params.get('backbone', 'efficientnet-b4'),
                    dropout_rate=params.get('dropout_rate', 0.3)
                )
                
                # Create data loaders
                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=params.get('batch_size', 16),
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                
                val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=params.get('batch_size', 16),
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Create trainer
                trainer = ClassificationTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=self.device,
                    use_wandb=False
                )
                
                # Update optimizer
                trainer.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=params.get('learning_rate', 1e-4),
                    weight_decay=params.get('weight_decay', 1e-4)
                )
                
                # Train for specified epochs
                best_val_f1 = 0.0
                for epoch in range(epochs):
                    trainer.train_epoch()
                    val_metrics = trainer.validate_epoch()
                    val_f1 = val_metrics.get('macro_f1', 0.0)
                    best_val_f1 = max(best_val_f1, val_f1)
                
                # Store results
                result = {
                    'params': params,
                    'score': best_val_f1,
                    'trial': i
                }
                results.append(result)
                
                # Update best
                if best_val_f1 > best_score:
                    best_score = best_val_f1
                    best_params = params
                
                logger.info(f"Score: {best_val_f1:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {i+1} failed: {str(e)}")
                results.append({
                    'params': params,
                    'score': 0.0,
                    'trial': i,
                    'error': str(e)
                })
        
        # Save results
        final_results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'n_trials': len(param_combinations)
        }
        
        results_path = save_dir / "grid_search_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Grid search completed. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return final_results

# Example usage
if __name__ == "__main__":
    # Example parameter grid for grid search
    param_grid = {
        'backbone': ['efficientnet-b2', 'efficientnet-b4'],
        'learning_rate': [1e-4, 1e-3],
        'batch_size': [16, 32],
        'dropout_rate': [0.2, 0.3, 0.4]
    }
    
    print("Hyperparameter tuning utilities ready!")
    print("Use HyperparameterTuner for Optuna-based optimization")
    print("Use GridSearchTuner for grid search optimization")
