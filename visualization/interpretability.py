"""
Interpretability and visualization tools for VinBigData Chest X-ray models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GradCAM:
    """
    Grad-CAM implementation for model interpretability
    """
    
    def __init__(self, model: nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations[id(module)] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients[id(module)] = grad_output[0].detach()
        
        # Register hooks for target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                hook_f = module.register_forward_hook(forward_hook)
                hook_b = module.register_backward_hook(backward_hook)
                self.hooks.extend([hook_f, hook_b])
    
    def generate_cam(self, 
                    input_tensor: torch.Tensor,
                    target_class: int,
                    class_names: List[str]) -> np.ndarray:
        """Generate Grad-CAM for target class"""
        
        # Forward pass
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        logits = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        target = torch.zeros_like(logits)
        target[0, target_class] = 1.0
        logits.backward(gradient=target, retain_graph=True)
        
        # Get gradients and activations for the last target layer
        last_layer_id = None
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                last_layer_id = id(module)
        
        if last_layer_id is None:
            raise ValueError("No target layers found")
        
        gradients = self.gradients[last_layer_id]
        activations = self.activations[last_layer_id]
        
        # Compute weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize_cam(self,
                     image: np.ndarray,
                     cam: np.ndarray,
                     target_class: int,
                     class_names: List[str],
                     alpha: float = 0.4,
                     save_path: Optional[str] = None) -> np.ndarray:
        """Visualize Grad-CAM overlay"""
        
        # Create colormap
        colormap = plt.cm.jet
        cam_colored = colormap(cam)[:, :, :3]
        cam_colored = (cam_colored * 255).astype(np.uint8)
        
        # Resize CAM to match image
        if cam.shape != image.shape[:2]:
            cam_colored = cv2.resize(cam_colored, (image.shape[1], image.shape[0]))
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
        
        # Add text
        class_name = class_names[target_class]
        cv2.putText(overlay, f"Grad-CAM: {class_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, overlay)
            logger.info(f"Grad-CAM visualization saved to {save_path}")
        
        return overlay
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class AttentionVisualizer:
    """
    Visualize attention maps from transformer-based models
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_maps = {}
        self.hooks = []
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights"""
        def attention_hook(module, input, output):
            if hasattr(module, 'attention_weights'):
                self.attention_maps[id(module)] = module.attention_weights.detach()
        
        # Register hooks for attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(attention_hook)
                self.hooks.append(hook)
    
    def get_attention_maps(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attention maps for input"""
        self.attention_maps.clear()
        self._register_attention_hooks()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        self._remove_hooks()
        return self.attention_maps
    
    def _remove_hooks(self):
        """Remove attention hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class ModelInterpretability:
    """
    Main class for model interpretability analysis
    """
    
    def __init__(self, 
                 model: nn.Module,
                 class_names: List[str],
                 device: str = "cuda"):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.model.eval()
    
    def analyze_prediction(self,
                          image: np.ndarray,
                          prediction: torch.Tensor,
                          target_classes: Optional[List[int]] = None,
                          save_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Analyze a single prediction with multiple interpretability methods"""
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert image to tensor
        if len(image.shape) == 3:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        else:
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        image_tensor = image_tensor.to(self.device)
        
        # Get prediction probabilities
        probs = torch.sigmoid(prediction).cpu().numpy()
        
        # Determine target classes
        if target_classes is None:
            target_classes = np.argsort(probs[0])[-3:][::-1]  # Top 3 predictions
        
        results = {}
        
        # Generate Grad-CAM for each target class
        try:
            grad_cam = GradCAM(self.model, ['backbone.features.8'])  # Example layer
            
            for class_idx in target_classes:
                if class_idx < len(self.class_names):
                    cam = grad_cam.generate_cam(image_tensor, class_idx, self.class_names)
                    
                    # Visualize
                    overlay = grad_cam.visualize_cam(
                        image, cam, class_idx, self.class_names
                    )
                    
                    results[f'gradcam_class_{class_idx}'] = overlay
                    
                    if save_dir:
                        save_path = save_dir / f"gradcam_class_{class_idx}_{self.class_names[class_idx]}.png"
                        cv2.imwrite(str(save_path), overlay)
            
            grad_cam.remove_hooks()
            
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {str(e)}")
        
        return results
    
    def create_prediction_visualization(self,
                                      image: np.ndarray,
                                      predictions: np.ndarray,
                                      ground_truth: Optional[np.ndarray] = None,
                                      save_path: Optional[str] = None) -> np.ndarray:
        """Create comprehensive prediction visualization"""
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Prediction probabilities
        class_names_short = [name.split()[0] for name in self.class_names]
        y_pos = np.arange(len(class_names_short))
        
        axes[0, 1].barh(y_pos, predictions)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(class_names_short)
        axes[0, 1].set_xlabel('Probability')
        axes[0, 1].set_title('Prediction Probabilities')
        
        # Ground truth comparison
        if ground_truth is not None:
            axes[1, 0].barh(y_pos, ground_truth, alpha=0.7, label='Ground Truth', color='green')
            axes[1, 0].barh(y_pos, predictions, alpha=0.7, label='Predictions', color='red')
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(class_names_short)
            axes[1, 0].set_xlabel('Probability')
            axes[1, 0].set_title('Ground Truth vs Predictions')
            axes[1, 0].legend()
        else:
            axes[1, 0].axis('off')
        
        # Confidence scores
        confidence_scores = predictions
        axes[1, 1].bar(range(len(confidence_scores)), confidence_scores)
        axes[1, 1].set_xlabel('Class Index')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_title('Confidence Scores')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction visualization saved to {save_path}")
        
        return fig
    
    def create_class_activation_maps(self,
                                   image: np.ndarray,
                                   model: nn.Module,
                                   target_layers: List[str],
                                   save_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Create class activation maps for multiple classes"""
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert image to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Get top predictions
        top_classes = np.argsort(probs)[-5:][::-1]
        
        results = {}
        
        try:
            grad_cam = GradCAM(model, target_layers)
            
            for class_idx in top_classes:
                if probs[class_idx] > 0.1:  # Only for confident predictions
                    cam = grad_cam.generate_cam(image_tensor, class_idx, self.class_names)
                    
                    # Create visualization
                    overlay = grad_cam.visualize_cam(
                        image, cam, class_idx, self.class_names
                    )
                    
                    results[f'cam_class_{class_idx}'] = overlay
                    
                    if save_dir:
                        save_path = save_dir / f"cam_class_{class_idx}_{self.class_names[class_idx]}.png"
                        cv2.imwrite(str(save_path), overlay)
            
            grad_cam.remove_hooks()
            
        except Exception as e:
            logger.error(f"CAM generation failed: {str(e)}")
        
        return results

def create_interpretability_report(model: nn.Module,
                                 test_dataset,
                                 class_names: List[str],
                                 num_samples: int = 10,
                                 save_dir: str = "interpretability_report") -> Dict[str, Any]:
    """Create comprehensive interpretability report"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    interpreter = ModelInterpretability(model, class_names)
    
    report = {
        'samples': [],
        'summary': {}
    }
    
    logger.info(f"Creating interpretability report with {num_samples} samples...")
    
    for i in range(min(num_samples, len(test_dataset))):
        # Get sample
        image, target = test_dataset[i]
        
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            if image_np.shape[2] == 1:
                image_np = image_np.squeeze(2)
        else:
            image_np = image
        
        # Get prediction
        with torch.no_grad():
            if isinstance(image, torch.Tensor):
                pred = model(image.unsqueeze(0))
            else:
                pred = model(torch.from_numpy(image).unsqueeze(0))
        
        pred_probs = torch.sigmoid(pred).cpu().numpy()[0]
        
        # Analyze prediction
        sample_dir = save_dir / f"sample_{i}"
        sample_dir.mkdir(exist_ok=True)
        
        # Create visualizations
        try:
            # Prediction visualization
            pred_viz = interpreter.create_prediction_visualization(
                image_np, pred_probs, target.numpy() if isinstance(target, torch.Tensor) else target
            )
            pred_viz.savefig(sample_dir / "prediction_analysis.png")
            plt.close(pred_viz)
            
            # CAM analysis
            cam_results = interpreter.create_class_activation_maps(
                image_np, model, ['backbone.features.8'], sample_dir
            )
            
            sample_report = {
                'sample_id': i,
                'predictions': pred_probs.tolist(),
                               'ground_truth': target.tolist() if isinstance(target, torch.Tensor) else target,
                'cam_results': list(cam_results.keys())
            }
            
            report['samples'].append(sample_report)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
    
    # Save report
    report_path = save_dir / "interpretability_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Interpretability report saved to {save_dir}")
    return report

# Example usage
if __name__ == "__main__":
    print("Interpretability tools ready!")
    print("Use GradCAM for gradient-based attention visualization")
    print("Use ModelInterpretability for comprehensive analysis")
