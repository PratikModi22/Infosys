"""
Detection model for VinBigData Chest X-ray abnormalities
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ChestXrayDetector:
    """
    Object detection model for chest X-ray abnormalities using YOLOv8
    """
    
    def __init__(self, 
                 model_size: str = "yolov8n",
                 num_classes: int = 14,
                 input_size: Tuple[int, int] = (640, 640),
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        
        self.model_size = model_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize YOLO model
        self.model = YOLO(f"{model_size}.pt")
        
        # Update model configuration
        self.model.conf = confidence_threshold
        self.model.iou = iou_threshold
        
        logger.info(f"Initialized {model_size} model for {num_classes} classes")
    
    def train(self, 
              data_config: str,
              epochs: int = 100,
              batch_size: int = 8,
              learning_rate: float = 1e-3,
              device: str = "cuda",
              save_dir: str = "runs/detect/train",
              **kwargs) -> Dict:
        """
        Train the detection model
        """
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Training parameters
        train_params = {
            'data': data_config,
            'epochs': epochs,
            'batch': batch_size,
            'lr0': learning_rate,
            'device': device,
            'project': save_dir,
            'name': 'chest_xray_detection',
            'save': True,
            'save_period': 10,
            'patience': 20,
            'imgsz': self.input_size[0],
            'conf': self.confidence_threshold,
            'iou': self.iou_threshold,
            'augment': True,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'val': True,
            'plots': True,
            'verbose': True
        }
        
        # Update with additional parameters
        train_params.update(kwargs)
        
        # Start training
        results = self.model.train(**train_params)
        
        logger.info("Training completed!")
        return results
    
    def validate(self, data_config: str, **kwargs) -> Dict:
        """
        Validate the model
        """
        logger.info("Starting validation...")
        
        val_params = {
            'data': data_config,
            'imgsz': self.input_size[0],
            'conf': self.confidence_threshold,
            'iou': self.iou_threshold,
            'device': kwargs.get('device', 'cuda'),
            'plots': True,
            'save_json': True,
            'verbose': True
        }
        
        val_params.update(kwargs)
        
        results = self.model.val(**val_params)
        
        logger.info("Validation completed!")
        return results
    
    def predict(self, 
                source: Union[str, np.ndarray, torch.Tensor],
                save: bool = False,
                save_dir: str = "runs/detect/predict",
                **kwargs) -> List[Dict]:
        """
        Make predictions on input data
        """
        predict_params = {
            'source': source,
            'conf': self.confidence_threshold,
            'iou': self.iou_threshold,
            'imgsz': self.input_size[0],
            'save': save,
            'project': save_dir,
            'name': 'predictions',
            'verbose': False
        }
        
        predict_params.update(kwargs)
        
        results = self.model.predict(**predict_params)
        
        # Convert results to standardized format
        predictions = []
        for result in results:
            pred_dict = {
                'image_path': result.path,
                'boxes': [],
                'scores': [],
                'class_ids': [],
                'class_names': []
            }
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                pred_dict['boxes'] = boxes.tolist()
                pred_dict['scores'] = scores.tolist()
                pred_dict['class_ids'] = class_ids.tolist()
                
                # Map class IDs to names (assuming standard class names)
                class_names = self._get_class_names()
                pred_dict['class_names'] = [class_names[cls_id] for cls_id in class_ids]
            
            predictions.append(pred_dict)
        
        return predictions
    
    def _get_class_names(self) -> List[str]:
        """Get class names for the model"""
        # This should match the class names from your dataset
        return [
            "Aortic enlargement",
            "Atelectasis", 
            "Calcification",
            "Cardiomegaly",
            "Consolidation",
            "ILD",
            "Infiltration",
            "Lung Opacity",
            "Nodule/Mass",
            "Other lesion",
            "Pleural effusion",
            "Pleural thickening",
            "Pneumothorax",
            "Pulmonary fibrosis"
        ]
    
    def load_weights(self, weights_path: str):
        """Load pretrained weights"""
        self.model = YOLO(weights_path)
        logger.info(f"Loaded weights from {weights_path}")
    
    def save_weights(self, save_path: str):
        """Save model weights"""
        self.model.save(save_path)
        logger.info(f"Saved weights to {save_path}")

class DetectionMetrics:
    """
    Metrics calculator for object detection
    """
    
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
    
    def update(self, pred_boxes: List[np.ndarray], 
               pred_scores: List[np.ndarray],
               pred_classes: List[np.ndarray],
               target_boxes: List[np.ndarray],
               target_classes: List[np.ndarray]):
        """Update metrics with batch predictions"""
        self.predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'classes': pred_classes
        })
        self.targets.append({
            'boxes': target_boxes,
            'classes': target_classes
        })
    
    def compute(self) -> Dict[str, float]:
        """Compute detection metrics"""
        if not self.predictions:
            return {}
        
        # This is a simplified implementation
        # In practice, you'd use more sophisticated metrics like mAP
        metrics = {}
        
        # Calculate average precision for each class
        for class_id in range(self.num_classes):
            # Simplified AP calculation
            # In practice, use proper COCO evaluation
            metrics[f'AP_class_{class_id}'] = 0.0
        
        # Overall mAP
        metrics['mAP'] = 0.0
        metrics['mAP50'] = 0.0
        metrics['mAP75'] = 0.0
        
        return metrics

def create_yolo_data_config(data_dir: str, 
                           train_images: str,
                           val_images: str,
                           train_labels: str,
                           val_labels: str,
                           class_names: List[str],
                           output_path: str) -> str:
    """
    Create YOLO data configuration file
    """
    config = {
        'path': data_dir,
        'train': train_images,
        'val': val_images,
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created YOLO data config: {output_path}")
    return output_path

def visualize_predictions(image: np.ndarray,
                         boxes: List[List[float]],
                         scores: List[float],
                         class_ids: List[int],
                         class_names: List[str],
                         save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize detection predictions on image
    """
    vis_image = image.copy()
    
    # Define colors for different classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (192, 192, 192), (128, 128, 128)
    ]
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_names[class_id]}: {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(vis_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_image)
        logger.info(f"Saved visualization to {save_path}")
    
    return vis_image

# Example usage
if __name__ == "__main__":
    # Test detector creation
    detector = ChestXrayDetector(
        model_size="yolov8n",
        num_classes=14,
        input_size=(640, 640)
    )
    
    print(f"Detector created: {detector.model_size}")
    
    # Test prediction on dummy image
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    predictions = detector.predict(dummy_image)
    print(f"Predictions: {len(predictions)}")
