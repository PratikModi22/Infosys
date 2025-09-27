"""
Classification model for VinBigData Chest X-ray abnormalities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ChestXrayClassifier(nn.Module):
    """
    Multi-label classification model for chest X-ray abnormalities
    """
    
    def __init__(self, 
                 num_classes: int = 14,
                 backbone: str = "efficientnet-b4",
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        super(ChestXrayClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load backbone
        if backbone.startswith("efficientnet"):
            self.backbone = timm.create_model(backbone, pretrained=pretrained)
            # Get number of features from backbone
            self.feature_dim = self.backbone.classifier.in_features
            # Remove the original classifier
            self.backbone.classifier = nn.Identity()
        elif backbone.startswith("resnet"):
            self.backbone = models.__dict__[backbone](pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps for visualization"""
        # This would be implemented for Grad-CAM
        # For now, return the feature maps from the last conv layer
        if hasattr(self.backbone, 'get_attention_maps'):
            return self.backbone.get_attention_maps(x)
        else:
            # Fallback: return the input for now
            return x

class MultiLabelLoss(nn.Module):
    """
    Custom loss function for multi-label classification
    """
    
    def __init__(self, 
                 pos_weight: Optional[torch.Tensor] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super(MultiLabelLoss, self).__init__()
        
        self.pos_weight = pos_weight
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Base BCE loss
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            weight=class_weights,
            reduction='none'
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for multi-label classification"""
        # Compute BCE loss
        bce_loss = self.bce_loss(logits, targets)
        
        # Compute focal loss
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma
        
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

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
    """Create model from configuration"""
    model = ChestXrayClassifier(
        num_classes=config['num_classes'],
        backbone=config['backbone'],
        pretrained=config.get('pretrained', True),
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    
    return model

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
        'backbone': 'efficientnet-b4',
        'pretrained': True,
        'dropout_rate': 0.3
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
