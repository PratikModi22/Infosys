"""
Data utilities for preparing the VinBigData Chest X-ray dataset.

What this file does:
- Creates a smaller subset of the dataset (default 2GB) for fast training
- Provides a simple DICOM→PNG converter
- Converts annotations from CSV to YOLO and COCO formats
- Creates train/val/test CSV splits
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Optional
import logging
from tqdm import tqdm
import json

from config import DATASET_CONFIG, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

class VinBigDataProcessor:
    """
    Main helper for data prep.

    Responsibilities:
    - Manage paths and outputs under `data/processed`
    - Build a size-limited subset by sampling images per class
    - Offer simple image and annotation conversion utilities
    - Create CSV splits for train/val/test
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.processed_dir = PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def create_subset(self, target_size_gb: float = 2.0) -> List[str]:
        """
        Create a subset (default ~2GB) by sampling images from each class.

        Parameters:
        - target_size_gb: approximate desired size on disk in gigabytes.

        Returns:
        - List of selected image_ids included in the subset.
        """
        logger.info(f"Creating {target_size_gb}GB subset of the dataset...")
        
        # Load metadata
        train_df = pd.read_csv(self.data_dir / "train.csv")
        
        # Calculate file sizes and create subset
        selected_files = []
        current_size = 0
        target_size_bytes = target_size_gb * 1024**3
        
        # Group by class to ensure diversity
        class_groups = train_df.groupby('class_name')
        
        for class_name, group in class_groups:
            if current_size >= target_size_bytes:
                break
                
            # Sample a small portion from each class to keep diversity
            sample_size = min(len(group), max(1, len(group) // 10))
            sampled = group.sample(n=sample_size, random_state=42)
            
            for _, row in sampled.iterrows():
                if current_size >= target_size_bytes:
                    break
                    
                # Estimate per-image size to accumulate toward target (rough DICOM avg)
                estimated_size = 2 * 1024 * 1024  # ~2MB estimate
                
                if current_size + estimated_size <= target_size_bytes:
                    selected_files.append(row['image_id'])
                    current_size += estimated_size
        
        # Save subset metadata
        subset_df = train_df[train_df['image_id'].isin(selected_files)]
        subset_df.to_csv(self.processed_dir / "subset_metadata.csv", index=False)
        
        logger.info(f"Created subset with {len(selected_files)} files (~{current_size/1024**3:.2f}GB)")
        return selected_files
    
    def convert_dicom_to_png(self, dicom_path: str, output_path: str, 
                           target_size: Tuple[int, int] = (512, 512)) -> bool:
        """
        Convert a single DICOM file to an 8-bit PNG.

        Steps:
        - Read pixel data, apply windowing if present
        - Normalize to 0–255, resize to target, and save as PNG
        Returns True on success, False otherwise.
        """
        try:
            # Read DICOM file
            dicom = pydicom.dcmread(dicom_path)
            
            # Extract pixel array
            pixel_array = dicom.pixel_array
            
            # Apply windowing if available
            if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
                window_center = dicom.WindowCenter
                window_width = dicom.WindowWidth
                
                # Apply windowing
                pixel_array = self.apply_windowing(pixel_array, window_center, window_width)
            
            # Normalize to 0-255
            pixel_array = self.normalize_image(pixel_array)
            
            # Resize to target size
            pixel_array = cv2.resize(pixel_array, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to PIL Image and save
            image = Image.fromarray(pixel_array.astype(np.uint8))
            image.save(output_path, 'PNG', optimize=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting {dicom_path}: {str(e)}")
            return False
    
    def apply_windowing(self, image: np.ndarray, window_center: float, 
                       window_width: float) -> np.ndarray:
        """
        Apply DICOM windowing to enhance contrast.
        """
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        # Clip values to window
        image = np.clip(image, window_min, window_max)
        
        # Normalize to 0-1
        image = (image - window_min) / (window_max - window_min)
        
        return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0–255 (uint8) for PNG saving.
        """
        # Handle different bit depths
        if image.dtype == np.uint16:
            # Convert 16-bit to 8-bit
            image = (image / 256).astype(np.uint8)
        elif image.dtype == np.uint8:
            pass  # Already 8-bit
        else:
            # Normalize to 0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        return image
    
    def convert_annotations_to_yolo(self, csv_path: str, output_dir: str) -> bool:
        """
        Convert annotations (CSV) to YOLO txt files.

        One file per image with lines: `class_id cx cy w h` (normalized).
        """
        try:
            df = pd.read_csv(csv_path)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create class mapping
            class_names = DATASET_CONFIG["class_names"]
            class_to_id = {name: idx for idx, name in enumerate(class_names)}
            
            # Group by image
            for image_id, group in df.groupby('image_id'):
                # Create YOLO annotation file
                yolo_file = output_dir / f"{image_id}.txt"
                
                with open(yolo_file, 'w') as f:
                    for _, row in group.iterrows():
                        if pd.isna(row['class_name']):
                            continue
                            
                        # Convert bounding box to YOLO format
                        x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
                        
                        # Get image dimensions (assuming 512x512 from preprocessing)
                        img_width, img_height = 512, 512
                        
                        # Convert to YOLO format (normalized center coordinates and dimensions)
                        center_x = (x_min + x_max) / 2 / img_width
                        center_y = (y_min + y_max) / 2 / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        
                        # Get class ID
                        class_id = class_to_id.get(row['class_name'], -1)
                        if class_id == -1:
                            continue
                        
                        # Write YOLO format: class_id center_x center_y width height
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            logger.info(f"Converted annotations to YOLO format in {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting annotations: {str(e)}")
            return False
    
    def convert_annotations_to_coco(self, csv_path: str, output_path: str) -> bool:
        """
        Convert annotations (CSV) to a single COCO-style JSON file.
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Create COCO structure
            coco_data = {
                "images": [],
                "annotations": [],
                "categories": []
            }
            
            # Add categories
            for idx, class_name in enumerate(DATASET_CONFIG["class_names"]):
                coco_data["categories"].append({
                    "id": idx,
                    "name": class_name,
                    "supercategory": "abnormality"
                })
            
            # Process images and annotations
            image_id = 0
            annotation_id = 0
            
            for img_id, group in df.groupby('image_id'):
                # Add image info
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": f"{img_id}.png",
                    "width": 512,
                    "height": 512
                })
                
                # Add annotations
                for _, row in group.iterrows():
                    if pd.isna(row['class_name']):
                        continue
                    
                    class_id = DATASET_CONFIG["class_names"].index(row['class_name'])
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [row['x_min'], row['y_min'], 
                                row['x_max'] - row['x_min'], 
                                row['y_max'] - row['y_min']],
                        "area": (row['x_max'] - row['x_min']) * (row['y_max'] - row['y_min']),
                        "iscrowd": 0
                    })
                    
                    annotation_id += 1
                
                image_id += 1
            
            # Save COCO format
            with open(output_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            logger.info(f"Converted annotations to COCO format: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting to COCO format: {str(e)}")
            return False
    
    def create_data_splits(self, metadata_path: str, train_ratio: float = 0.7, 
                          val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, List[str]]:
        """
        Create simple train/val/test splits by shuffling unique image_ids.
        """
        df = pd.read_csv(metadata_path)
        
        # Get unique image IDs
        unique_images = df['image_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_images)
        
        # Calculate split sizes
        n_total = len(unique_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Create splits
        splits = {
            'train': unique_images[:n_train],
            'val': unique_images[n_train:n_train + n_val],
            'test': unique_images[n_train + n_val:]
        }
        
        # Save splits
        for split_name, image_ids in splits.items():
            split_df = df[df['image_id'].isin(image_ids)]
            split_df.to_csv(self.processed_dir / f"{split_name}_split.csv", index=False)
        
        logger.info(f"Created data splits - Train: {len(splits['train'])}, "
                   f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return splits

def main():
    """Command-line entry for running basic data prep steps."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VinBigData Data Preparation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--subset_size", type=float, default=2.0, help="Subset size in GB")
    parser.add_argument("--convert_images", action="store_true", help="Convert DICOM to PNG")
    parser.add_argument("--convert_annotations", action="store_true", help="Convert annotations")
    parser.add_argument("--create_splits", action="store_true", help="Create train/val/test splits")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = VinBigDataProcessor(args.data_dir)
    
    # Create subset
    if args.subset_size > 0:
        selected_files = processor.create_subset(args.subset_size)
        print(f"Created subset with {len(selected_files)} files")
    
    # Convert images
    if args.convert_images:
        print("Converting DICOM images to PNG...")
        # Implementation for batch conversion would go here
    
    # Convert annotations
    if args.convert_annotations:
        print("Converting annotations...")
        metadata_path = processor.processed_dir / "subset_metadata.csv"
        processor.convert_annotations_to_yolo(metadata_path, processor.processed_dir / "yolo_annotations")
        processor.convert_annotations_to_coco(metadata_path, processor.processed_dir / "coco_annotations.json")
    
    # Create splits
    if args.create_splits:
        print("Creating data splits...")
        metadata_path = processor.processed_dir / "subset_metadata.csv"
        processor.create_data_splits(metadata_path)

if __name__ == "__main__":
    main()
