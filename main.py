"""
Main script for VinBigData Chest X-ray project
"""
import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from data_preparation.data_utils import VinBigDataProcessor
from data_preparation.dataset import create_classification_datasets, create_detection_datasets
from models.classification_model import create_model as create_classification_model
from models.detection_model import ChestXrayDetector, create_yolo_data_config
from training.trainer import ClassificationTrainer, DetectionTrainer, create_data_loaders
from optimization.hyperparameter_tuning import HyperparameterTuner, GridSearchTuner
from visualization.interpretability import ModelInterpretability, create_interpretability_report

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('chest_xray_project.log')
        ]
    )

def data_preparation_pipeline(args):
    """Run data preparation pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data preparation pipeline...")
    
    # Initialize processor
    processor = VinBigDataProcessor(args.data_dir)
    
    # Create 5GB subset
    if args.subset_size > 0:
        logger.info(f"Creating {args.subset_size}GB subset...")
        selected_files = processor.create_5gb_subset(args.subset_size)
        logger.info(f"Created subset with {len(selected_files)} files")
    
    # Convert DICOM to PNG
    if args.convert_images:
        logger.info("Converting DICOM images to PNG...")
        # This would be implemented with batch processing
        # processor.convert_dicom_batch(args.data_dir, args.output_dir)
    
    # Convert annotations
    if args.convert_annotations:
        logger.info("Converting annotations...")
        metadata_path = processor.processed_dir / "subset_metadata.csv"
        processor.convert_annotations_to_yolo(metadata_path, processor.processed_dir / "yolo_annotations")
        processor.convert_annotations_to_coco(metadata_path, processor.processed_dir / "coco_annotations.json")
    
    # Create data splits
    if args.create_splits:
        logger.info("Creating data splits...")
        metadata_path = processor.processed_dir / "subset_metadata.csv"
        processor.create_data_splits(metadata_path)
    
    logger.info("Data preparation completed!")

def classification_training_pipeline(args):
    """Run classification training pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting classification training...")
    
    # Create datasets
    datasets = create_classification_datasets(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        class_names=DATASET_CONFIG["class_names"],
        image_size=MODEL_CONFIG['classification']['input_size']
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        datasets['train'], 
        datasets['val'],
        batch_size=MODEL_CONFIG['classification']['batch_size']
    )
    
    # Create model
    model = create_classification_model(MODEL_CONFIG['classification'])
    
    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        use_wandb=args.use_wandb
    )
    
    # Train model
    results = trainer.train(
        epochs=args.epochs,
        save_dir=args.save_dir,
        early_stopping_patience=args.patience
    )
    
    logger.info("Classification training completed!")

def detection_training_pipeline(args):
    """Run detection training pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting detection training...")
    
    # Create YOLO data config
    data_config_path = create_yolo_data_config(
        data_dir=args.data_dir,
        train_images="train/images",
        val_images="val/images", 
        train_labels="train/labels",
        val_labels="val/labels",
        class_names=DATASET_CONFIG["class_names"],
        output_path="yolo_data.yaml"
    )
    
    # Create detector
    detector = ChestXrayDetector(
        model_size=args.model_size,
        num_classes=len(DATASET_CONFIG["class_names"]),
        input_size=MODEL_CONFIG['detection']['input_size']
    )
    
    # Create trainer
    trainer = DetectionTrainer(
        model=detector,
        data_config=data_config_path,
        device=args.device,
        use_wandb=args.use_wandb
    )
    
    # Train model
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    logger.info("Detection training completed!")

def hyperparameter_tuning_pipeline(args):
    """Run hyperparameter tuning pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting hyperparameter tuning...")
    
    # Create datasets
    datasets = create_classification_datasets(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        class_names=DATASET_CONFIG["class_names"]
    )
    
    if args.method == "optuna":
        # Optuna-based tuning
        tuner = HyperparameterTuner(
            train_dataset=datasets['train'],
            val_dataset=datasets['val'],
            class_names=DATASET_CONFIG["class_names"],
            device=args.device,
            n_trials=args.n_trials,
            timeout=args.timeout
        )
        
        results = tuner.optimize(save_dir=args.save_dir)
        
    elif args.method == "grid":
        # Grid search tuning
        param_grid = {
            'backbone': ['efficientnet-b2', 'efficientnet-b4'],
            'learning_rate': [1e-4, 1e-3],
            'batch_size': [16, 32],
            'dropout_rate': [0.2, 0.3, 0.4]
        }
        
        tuner = GridSearchTuner(
            train_dataset=datasets['train'],
            val_dataset=datasets['val'],
            class_names=DATASET_CONFIG["class_names"],
            device=args.device
        )
        
        results = tuner.grid_search(
            param_grid=param_grid,
            epochs=args.epochs,
            save_dir=args.save_dir
        )
    
    logger.info("Hyperparameter tuning completed!")

def interpretability_pipeline(args):
    """Run interpretability analysis pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting interpretability analysis...")
    
    # Load model
    model = create_classification_model(MODEL_CONFIG['classification'])
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create datasets
    datasets = create_classification_datasets(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        class_names=DATASET_CONFIG["class_names"]
    )
    
    # Create interpretability report
    report = create_interpretability_report(
        model=model,
        test_dataset=datasets['test'],
        class_names=DATASET_CONFIG["class_names"],
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
    
    logger.info("Interpretability analysis completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="VinBigData Chest X-ray Project")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data preparation
    data_parser = subparsers.add_parser('data', help='Data preparation pipeline')
    data_parser.add_argument('--data_dir', type=str, required=True, help='Path to raw dataset')
    data_parser.add_argument('--subset_size', type=float, default=5.0, help='Subset size in GB')
    data_parser.add_argument('--convert_images', action='store_true', help='Convert DICOM to PNG')
    data_parser.add_argument('--convert_annotations', action='store_true', help='Convert annotations')
    data_parser.add_argument('--create_splits', action='store_true', help='Create train/val/test splits')
    
    # Classification training
    cls_parser = subparsers.add_parser('train_classification', help='Train classification model')
    cls_parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data')
    cls_parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata CSV')
    cls_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    cls_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    cls_parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    cls_parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    cls_parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    cls_parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    # Detection training
    det_parser = subparsers.add_parser('train_detection', help='Train detection model')
    det_parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data')
    det_parser.add_argument('--model_size', type=str, default='yolov8n', help='YOLO model size')
    det_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    det_parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    det_parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    det_parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    det_parser.add_argument('--save_dir', type=str, default='runs/detect/train', help='Directory to save models')
    det_parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    # Hyperparameter tuning
    tune_parser = subparsers.add_parser('tune', help='Hyperparameter tuning')
    tune_parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data')
    tune_parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata CSV')
    tune_parser.add_argument('--method', type=str, choices=['optuna', 'grid'], default='optuna', help='Tuning method')
    tune_parser.add_argument('--n_trials', type=int, default=50, help='Number of trials (Optuna)')
    tune_parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    tune_parser.add_argument('--epochs', type=int, default=10, help='Epochs per trial')
    tune_parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    tune_parser.add_argument('--save_dir', type=str, default='hyperparameter_tuning', help='Directory to save results')
    
    # Interpretability
    interp_parser = subparsers.add_parser('interpret', help='Model interpretability analysis')
    interp_parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data')
    interp_parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata CSV')
    interp_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    interp_parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to analyze')
    interp_parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    interp_parser.add_argument('--save_dir', type=str, default='interpretability_report', help='Directory to save results')
    
    # Global arguments
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # Run appropriate pipeline
    if args.command == 'data':
        data_preparation_pipeline(args)
    elif args.command == 'train_classification':
        classification_training_pipeline(args)
    elif args.command == 'train_detection':
        detection_training_pipeline(args)
    elif args.command == 'tune':
        hyperparameter_tuning_pipeline(args)
    elif args.command == 'interpret':
        interpretability_pipeline(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
