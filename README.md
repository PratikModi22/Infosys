# VinBigData Chest X-ray Abnormalities Detection

A comprehensive AIML project for detecting abnormalities in chest X-ray images using deep learning models.

## Project Overview

This project implements both classification and detection models for the VinBigData Chest X-ray dataset, focusing on a 5GB subset for efficient development and testing.

### Key Features

- **Multi-label Classification**: EfficientNet/ResNet-based models for abnormality classification
- **Object Detection**: YOLOv8-based models for bounding box detection
- **Data Processing**: DICOM to PNG conversion with preprocessing
- **Hyperparameter Tuning**: Optuna and Grid Search optimization
- **Model Interpretability**: Grad-CAM and attention visualization
- **Comprehensive Evaluation**: Multiple metrics and visualization tools

## Dataset

The project uses the VinBigData Chest X-ray dataset with 14 abnormality classes:

1. Aortic enlargement
2. Atelectasis
3. Calcification
4. Cardiomegaly
5. Consolidation
6. ILD (Interstitial Lung Disease)
7. Infiltration
8. Lung Opacity
9. Nodule/Mass
10. Other lesion
11. Pleural effusion
12. Pleural thickening
13. Pneumothorax
14. Pulmonary fibrosis

## Project Structure

```
├── config.py                          # Configuration settings
├── main.py                           # Main entry point
├── requirements.txt                  # Python dependencies
├── data_preparation/
│   ├── data_utils.py                 # Data processing utilities
│   └── dataset.py                    # Dataset classes
├── models/
│   ├── classification_model.py       # Classification models
│   └── detection_model.py            # Detection models
├── training/
│   └── trainer.py                    # Training pipelines
├── optimization/
│   └── hyperparameter_tuning.py      # Hyperparameter optimization
├── visualization/
│   └── interpretability.py           # Model interpretability tools
└── README.md                         # This file
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd chest-xray-project
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup environment**:
```bash
# Create conda environment (recommended)
conda create -n chest-xray python=3.9
conda activate chest-xray
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Create a 5GB subset and prepare the data:

```bash
python main.py data \
    --data_dir /path/to/raw/dataset \
    --subset_size 5.0 \
    --convert_images \
    --convert_annotations \
    --create_splits
```

### 2. Classification Training

Train a classification model:

```bash
python main.py train_classification \
    --data_dir /path/to/processed/data \
    --metadata_path /path/to/metadata.csv \
    --epochs 100 \
    --batch_size 16 \
    --device cuda \
    --save_dir checkpoints/classification
```

### 3. Detection Training

Train a detection model:

```bash
python main.py train_detection \
    --data_dir /path/to/processed/data \
    --model_size yolov8n \
    --epochs 100 \
    --batch_size 8 \
    --device cuda \
    --save_dir runs/detect/train
```

### 4. Hyperparameter Tuning

Optimize hyperparameters using Optuna:

```bash
python main.py tune \
    --data_dir /path/to/processed/data \
    --metadata_path /path/to/metadata.csv \
    --method optuna \
    --n_trials 50 \
    --device cuda \
    --save_dir hyperparameter_tuning
```

### 5. Model Interpretability

Analyze model predictions and generate visualizations:

```bash
python main.py interpret \
    --data_dir /path/to/processed/data \
    --metadata_path /path/to/metadata.csv \
    --model_path /path/to/trained/model.pth \
    --num_samples 10 \
    --device cuda \
    --save_dir interpretability_report
```

## Configuration

The project uses a centralized configuration system in `config.py`:

- **Dataset Configuration**: Class names, image sizes, data paths
- **Model Configuration**: Architecture settings, hyperparameters
- **Training Configuration**: Training parameters, device settings
- **Augmentation Configuration**: Data augmentation settings

## Model Architectures

### Classification Models

- **EfficientNet**: EfficientNet-B0, B2, B4 variants
- **ResNet**: ResNet-50, ResNet-101
- **Custom Heads**: Multi-label classification with focal loss

### Detection Models

- **YOLOv8**: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- **Custom Configuration**: Optimized for medical imaging

## Evaluation Metrics

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC (Area Under Curve)
- Macro and Micro averages

### Detection Metrics
- mAP (mean Average Precision)
- mAP@0.5, mAP@0.75
- Precision, Recall per class

## Data Augmentation

The project includes comprehensive data augmentation:

- **Geometric**: Rotation, flipping, elastic transform
- **Photometric**: Brightness, contrast, hue adjustments
- **Noise**: Gaussian noise, blur effects
- **Advanced**: Mixup, CutMix (for detection)

## Hyperparameter Optimization

Two optimization methods are available:

1. **Optuna**: Bayesian optimization with TPE sampler
2. **Grid Search**: Exhaustive search over parameter grid

Optimized parameters include:
- Model architecture (backbone)
- Learning rate, batch size
- Dropout rate, weight decay
- Data augmentation strength

## Model Interpretability

Comprehensive interpretability tools:

- **Grad-CAM**: Gradient-weighted Class Activation Maps
- **Attention Visualization**: Transformer attention maps
- **Prediction Analysis**: Confidence scores and error analysis
- **Class Activation Maps**: Per-class attention visualization

## Results and Visualization

The project generates:

- **Training Curves**: Loss and metric plots
- **Confusion Matrices**: Classification performance
- **ROC Curves**: Per-class performance
- **Attention Maps**: Model focus visualization
- **Detection Results**: Bounding box visualizations

## Performance Optimization

- **Mixed Precision Training**: Faster training with FP16
- **Data Loading**: Optimized data loaders with multiple workers
- **Model Checkpointing**: Automatic saving of best models
- **Early Stopping**: Prevent overfitting

## Monitoring and Logging

- **Weights & Biases**: Experiment tracking and visualization
- **TensorBoard**: Training monitoring
- **Custom Logging**: Detailed training logs

## Future Enhancements

- [ ] Multi-task learning (classification + detection)
- [ ] Ensemble methods
- [ ] Advanced augmentation techniques
- [ ] Model compression and quantization
- [ ] Deployment optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VinBigData for providing the dataset
- PyTorch team for the deep learning framework
- Ultralytics for YOLOv8 implementation
- Optuna for hyperparameter optimization
- Weights & Biases for experiment tracking

## Contact

For questions or support, please open an issue in the repository.
