"""
Training pipeline for VinBigData Chest X-ray models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score

from models.classification_model import ChestXrayClassifier, MultiLabelLoss, ClassificationMetrics
from models.detection_model import ChestXrayDetector, DetectionMetrics
from config import TRAINING_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

class ClassificationTrainer:
    """
    Trainer for classification models
    """
    
    def __init__(self, 
                 model: ChestXrayClassifier,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda",
                 use_wandb: bool = False):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize loss and optimizer
        self.criterion = MultiLabelLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=MODEL_CONFIG['classification']['learning_rate'],
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Metrics
        self.train_metrics = ClassificationMetrics(
            num_classes=MODEL_CONFIG['classification']['num_classes']
        )
        self.val_metrics = ClassificationMetrics(
            num_classes=MODEL_CONFIG['classification']['num_classes']
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.patience_counter = 0
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="chest-xray-classification",
                config={
                    "model": MODEL_CONFIG['classification'],
                    "training": TRAINING_CONFIG
                }
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if TRAINING_CONFIG['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    TRAINING_CONFIG['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(logits.detach(), targets.detach())
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]['lr']
                })
        
        # Compute epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_metrics = self.train_metrics.compute()
        epoch_metrics['loss'] = epoch_loss
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                
                # Update metrics
                self.val_metrics.update(logits, targets)
                total_loss += loss.item()
        
        # Compute epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_metrics = self.val_metrics.compute()
        epoch_metrics['loss'] = epoch_loss
        
        return epoch_metrics
    
    def train(self, 
              epochs: int,
              save_dir: str = "checkpoints",
              early_stopping_patience: int = 10) -> Dict[str, Any]:
        """Train the model"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val AUC: {val_metrics.get('macro_f1', 0):.4f}")
            
            # Save metrics to history
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['train_metrics'].append(train_metrics)
            training_history['val_metrics'].append(val_metrics)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics['loss'],
                    "val/loss": val_metrics['loss'],
                    "val/macro_f1": val_metrics.get('macro_f1', 0),
                    "val/macro_precision": val_metrics.get('macro_precision', 0),
                    "val/macro_recall": val_metrics.get('macro_recall', 0)
                })
            
            # Save best model
            current_val_loss = val_metrics['loss']
            current_val_auc = val_metrics.get('macro_f1', 0)
            
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = save_dir / f"best_model_epoch_{epoch}.pth"
                self.save_checkpoint(checkpoint_path, val_metrics)
                logger.info(f"New best model saved: {checkpoint_path}")
            
            elif current_val_auc > self.best_val_auc:
                self.best_val_auc = current_val_auc
                self.patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = save_dir / f"best_auc_model_epoch_{epoch}.pth"
                self.save_checkpoint(checkpoint_path, val_metrics)
                logger.info(f"New best AUC model saved: {checkpoint_path}")
            
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        final_checkpoint_path = save_dir / "final_model.pth"
        self.save_checkpoint(final_checkpoint_path, val_metrics)
        
        # Save training history
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info("Training completed!")
        return training_history
    
    def save_checkpoint(self, save_path: str, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_auc': self.best_val_auc,
            'metrics': metrics
        }
        
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_auc = checkpoint['best_val_auc']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

class DetectionTrainer:
    """
    Trainer for detection models using YOLOv8
    """
    
    def __init__(self, 
                 model: ChestXrayDetector,
                 data_config: str,
                 device: str = "cuda",
                 use_wandb: bool = False):
        
        self.model = model
        self.data_config = data_config
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="chest-xray-detection",
                config={
                    "model": MODEL_CONFIG['detection'],
                    "training": TRAINING_CONFIG
                }
            )
    
    def train(self, 
              epochs: int,
              batch_size: int = 8,
              learning_rate: float = 1e-3,
              save_dir: str = "runs/detect/train",
              **kwargs) -> Dict[str, Any]:
        """Train the detection model"""
        
        logger.info(f"Starting detection training for {epochs} epochs...")
        
        # Training parameters
        train_params = {
            'data': self.data_config,
            'epochs': epochs,
            'batch': batch_size,
            'lr0': learning_rate,
            'device': self.device,
            'project': save_dir,
            'name': 'chest_xray_detection',
            'save': True,
            'save_period': 10,
            'patience': 20,
            'imgsz': MODEL_CONFIG['detection']['input_size'][0],
            'conf': MODEL_CONFIG['detection']['confidence_threshold'],
            'iou': MODEL_CONFIG['detection']['iou_threshold'],
            'augment': True,
            'plots': True,
            'verbose': True
        }
        
        # Update with additional parameters
        train_params.update(kwargs)
        
        # Start training
        results = self.model.train(**train_params)
        
        logger.info("Detection training completed!")
        return results
    
    def validate(self, **kwargs) -> Dict[str, Any]:
        """Validate the detection model"""
        logger.info("Starting detection validation...")
        
        val_params = {
            'data': self.data_config,
            'imgsz': MODEL_CONFIG['detection']['input_size'][0],
            'conf': MODEL_CONFIG['detection']['confidence_threshold'],
            'iou': MODEL_CONFIG['detection']['iou_threshold'],
            'device': self.device,
            'plots': True,
            'save_json': True,
            'verbose': True
        }
        
        val_params.update(kwargs)
        
        results = self.model.validate(**val_params)
        
        logger.info("Detection validation completed!")
        return results

def create_data_loaders(train_dataset, val_dataset, batch_size: int = 16, 
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # This would be used with actual datasets
    print("Training pipeline ready!")
    print("Use ClassificationTrainer for classification tasks")
    print("Use DetectionTrainer for detection tasks")
