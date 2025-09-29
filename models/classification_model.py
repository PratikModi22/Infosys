"""
Simple classification model for VinBigData Chest X-ray abnormalities.

What this file does:
- Provides a minimal, easy-to-understand multi-label classifier (ResNet18)
- Implements a simple focal-BCE loss for class imbalance
- Includes a tiny metrics helper for macro F1 etc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ChestXrayClassifier(nn.Module):
    """
    Minimal multi-label classifier using ResNet-18 as a feature extractor.

    Easy version:
    - ResNet18 backbone (pretrained on ImageNet)
    - Single linear layer head to 14 classes
    - Outputs raw logits for BCEWithLogits-based losses
    """
    
    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super(ChestXrayClassifier, self).__init__()
        
        backbone = models.resnet18(pretrained=pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        self.classifier = nn.Linear(in_features, num_classes)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for attention maps (used by Grad-CAM tools).
        Returns input as a simple fallback.
        """
        return x

class MultiLabelLoss(nn.Module):
    """
    Simple focal-BCE loss to handle class imbalance in multi-label tasks.
    """
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super(MultiLabelLoss, self).__init__()
        self.pos_weight = pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal-weighted BCE loss averaged over batch and classes."""
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = self.focal_alpha * (1 - pt) ** self.focal_gamma
        return (focal * bce).mean()

class ClassificationMetrics:
    """
    Metrics calculator for multi-label classification
    """
    
    def __init__(self, num_classes: int, threshold: float = 0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch predictions"""
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        self.predictions.append(preds.cpu())
        self.targets.append(targets.cpu())
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        if not self.predictions:
            return {}
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Compute metrics
        metrics = {}
        
        # Overall accuracy
        correct = (all_preds == all_targets).all(dim=1).float()
        metrics['accuracy'] = correct.mean().item()
        
        # Per-class metrics
        for i in range(self.num_classes):
            class_preds = all_preds[:, i]
            class_targets = all_targets[:, i]
            
            # Precision, Recall, F1
            tp = ((class_preds == 1) & (class_targets == 1)).sum().float()
            fp = ((class_preds == 1) & (class_targets == 0)).sum().float()
            fn = ((class_preds == 0) & (class_targets == 1)).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[f'class_{i}_precision'] = precision.item()
            metrics[f'class_{i}_recall'] = recall.item()
            metrics[f'class_{i}_f1'] = f1.item()
        
        # Macro averages
        precisions = [metrics[f'class_{i}_precision'] for i in range(self.num_classes)]
        recalls = [metrics[f'class_{i}_recall'] for i in range(self.num_classes)]
        f1s = [metrics[f'class_{i}_f1'] for i in range(self.num_classes)]
        
        metrics['macro_precision'] = sum(precisions) / len(precisions)
        metrics['macro_recall'] = sum(recalls) / len(recalls)
        metrics['macro_f1'] = sum(f1s) / len(f1s)
        
        return metrics

def create_model(config: Dict) -> ChestXrayClassifier:
    """
    Build a ResNet18-based classifier from a simple config dict.
    Expected keys: 'num_classes', optional 'pretrained'
    """
    return ChestXrayClassifier(
        num_classes=config['num_classes'],
        pretrained=config.get('pretrained', True)
    )

def load_pretrained_weights(model: ChestXrayClassifier, checkpoint_path: str) -> ChestXrayClassifier:
    """Load pretrained weights from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"Loaded pretrained weights from {checkpoint_path}")
    return model

def save_model_checkpoint(model: ChestXrayClassifier, 
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         loss: float,
                         metrics: Dict[str, float],
                         save_path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")

# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    config = {
        'num_classes': 14,
        'pretrained': True
    }
    
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    # Test metrics
    metrics = ClassificationMetrics(num_classes=14)
    targets = torch.randint(0, 2, (2, 14)).float()
    metrics.update(logits, targets)
    results = metrics.compute()
    print(f"Sample metrics: {results}")
