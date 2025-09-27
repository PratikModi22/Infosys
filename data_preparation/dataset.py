"""
Dataset classes for VinBigData Chest X-ray project
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)

class ChestXrayDataset(Dataset):
    """
    Dataset class for chest X-ray images with multi-label classification
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: List[List[int]],
                 class_names: List[str],
                 transform: Optional[A.Compose] = None,
                 image_size: Tuple[int, int] = (512, 512)):
        
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.image_size = image_size
        
        # Validate inputs
        assert len(image_paths) == len(labels), "Number of images and labels must match"
        assert len(class_names) == len(labels[0]), "Number of classes must match label dimensions"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        
        # Get labels
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Convert to tensor if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if image.shape[:2] != self.image_size:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a black image as fallback
            return np.zeros((*self.image_size, 3), dtype=np.uint8)

class ChestXrayDetectionDataset(Dataset):
    """
    Dataset class for chest X-ray images with object detection
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 annotations: List[Dict],
                 class_names: List[str],
                 transform: Optional[A.Compose] = None,
                 image_size: Tuple[int, int] = (640, 640)):
        
        self.image_paths = image_paths
        self.annotations = annotations
        self.class_names = class_names
        self.transform = transform
        self.image_size = image_size
        
        # Create class name to ID mapping
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Load image
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        
        # Get annotations
        annotation = self.annotations[idx]
        boxes = annotation['boxes']
        class_ids = annotation['class_ids']
        
        # Apply transforms
        if self.transform:
            # Prepare bboxes for albumentations
            bboxes = []
            for box, class_id in zip(boxes, class_ids):
                bboxes.append([*box, class_id])
            
            transformed = self.transform(image=image, bboxes=bboxes)
            image = transformed['image']
            
            # Extract transformed bboxes and class_ids
            transformed_bboxes = []
            transformed_class_ids = []
            for bbox in transformed['bboxes']:
                if len(bbox) >= 5:  # x1, y1, x2, y2, class_id
                    transformed_bboxes.append(bbox[:4])
                    transformed_class_ids.append(bbox[4])
            
            boxes = np.array(transformed_bboxes) if transformed_bboxes else np.array([])
            class_ids = np.array(transformed_class_ids) if transformed_class_ids else np.array([])
        else:
            # Convert to tensor if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes = np.array(boxes)
            class_ids = np.array(class_ids)
        
        # Prepare target dictionary
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(class_ids, dtype=torch.long),
            'image_id': torch.tensor(idx, dtype=torch.long),
            'area': torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes], dtype=torch.float32),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.long)
        }
        
        return image, target
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if image.shape[:2] != self.image_size:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a black image as fallback
            return np.zeros((*self.image_size, 3), dtype=np.uint8)

def get_augmentation_transforms(image_size: Tuple[int, int] = (512, 512), 
                              mode: str = "train") -> A.Compose:
    """
    Get augmentation transforms for training/validation
    """
    if mode == "train":
        transforms = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.1),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:  # validation/test
        transforms = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transforms

def create_classification_datasets(data_dir: str,
                                 metadata_path: str,
                                 class_names: List[str],
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 image_size: Tuple[int, int] = (512, 512)) -> Dict[str, ChestXrayDataset]:
    """
    Create train/val/test datasets for classification
    """
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Get unique images and their labels
    image_data = []
    for image_id, group in df.groupby('image_id'):
        # Create multi-label vector
        labels = [0] * len(class_names)
        for _, row in group.iterrows():
            if pd.notna(row['class_name']):
                class_idx = class_names.index(row['class_name'])
                labels[class_idx] = 1
        
        # Find corresponding image file
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = Path(data_dir) / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = str(potential_path)
                break
        
        if image_path:
            image_data.append((image_path, labels))
    
    # Shuffle data
    np.random.seed(42)
    np.random.shuffle(image_data)
    
    # Split data
    n_total = len(image_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = image_data[:n_train]
    val_data = image_data[n_train:n_train + n_val]
    test_data = image_data[n_train + n_val:]
    
    # Create datasets
    train_transform = get_augmentation_transforms(image_size, mode="train")
    val_transform = get_augmentation_transforms(image_size, mode="val")
    
    datasets = {
        'train': ChestXrayDataset(
            [item[0] for item in train_data],
            [item[1] for item in train_data],
            class_names,
            train_transform,
            image_size
        ),
        'val': ChestXrayDataset(
            [item[0] for item in val_data],
            [item[1] for item in val_data],
            class_names,
            val_transform,
            image_size
        ),
        'test': ChestXrayDataset(
            [item[0] for item in test_data],
            [item[1] for item in test_data],
            class_names,
            val_transform,
            image_size
        )
    }
    
    logger.info(f"Created datasets - Train: {len(datasets['train'])}, "
               f"Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    
    return datasets

def create_detection_datasets(data_dir: str,
                            metadata_path: str,
                            class_names: List[str],
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            image_size: Tuple[int, int] = (640, 640)) -> Dict[str, ChestXrayDetectionDataset]:
    """
    Create train/val/test datasets for detection
    """
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Get unique images and their annotations
    image_data = []
    for image_id, group in df.groupby('image_id'):
        # Collect bounding boxes and class IDs
        boxes = []
        class_ids = []
        
        for _, row in group.iterrows():
            if pd.notna(row['class_name']) and pd.notna(row['x_min']):
                # Convert to YOLO format (normalized coordinates)
                x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
                
                # Normalize coordinates (assuming 512x512 images)
                img_width, img_height = 512, 512
                x_min_norm = x_min / img_width
                y_min_norm = y_min / img_height
                x_max_norm = x_max / img_width
                y_max_norm = y_max / img_height
                
                boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
                class_ids.append(class_names.index(row['class_name']))
        
        # Find corresponding image file
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = Path(data_dir) / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = str(potential_path)
                break
        
        if image_path and boxes:
            annotation = {
                'boxes': boxes,
                'class_ids': class_ids
            }
            image_data.append((image_path, annotation))
    
    # Shuffle data
    np.random.seed(42)
    np.random.shuffle(image_data)
    
    # Split data
    n_total = len(image_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = image_data[:n_train]
    val_data = image_data[n_train:n_train + n_val]
    test_data = image_data[n_train + n_val:]
    
    # Create datasets
    train_transform = get_augmentation_transforms(image_size, mode="train")
    val_transform = get_augmentation_transforms(image_size, mode="val")
    
    datasets = {
        'train': ChestXrayDetectionDataset(
            [item[0] for item in train_data],
            [item[1] for item in train_data],
            class_names,
            train_transform,
            image_size
        ),
        'val': ChestXrayDetectionDataset(
            [item[0] for item in val_data],
            [item[1] for item in val_data],
            class_names,
            val_transform,
            image_size
        ),
        'test': ChestXrayDetectionDataset(
            [item[0] for item in test_data],
            [item[1] for item in test_data],
            class_names,
            val_transform,
            image_size
        )
    }
    
    logger.info(f"Created detection datasets - Train: {len(datasets['train'])}, "
               f"Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    
    return datasets

# Example usage
if __name__ == "__main__":
    # Test dataset creation
    class_names = [
        "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
        "Consolidation", "ILD", "Infiltration", "Lung Opacity",
        "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
        "Pneumothorax", "Pulmonary fibrosis"
    ]
    
    print("Dataset classes created successfully!")
    print(f"Number of classes: {len(class_names)}")
