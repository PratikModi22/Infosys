"""
Environment setup script for VinBigData Chest X-ray project
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8+ is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_cuda_availability():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available: {torch.cuda.get_device_name()}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA is not available, will use CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet, cannot check CUDA")
        return False

def install_requirements():
    """Install Python requirements"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python requirements"
    )

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "results",
        "logs",
        "checkpoints",
        "runs/detect/train",
        "runs/detect/val",
        "hyperparameter_tuning",
        "interpretability_report"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def setup_git_hooks():
    """Setup git hooks for code quality"""
    hooks_dir = Path(".git/hooks")
    if hooks_dir.exists():
        # Pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/sh
# Run code formatting
python -m black --check .
python -m flake8 .
"""
        with open(pre_commit_hook, 'w') as f:
            f.write(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        print("✓ Git pre-commit hook created")
    
    return True

def verify_installation():
    """Verify that all packages are installed correctly"""
    required_packages = [
        "torch", "torchvision", "numpy", "pandas", "opencv-python",
        "albumentations", "matplotlib", "seaborn", "tqdm", "optuna",
        "ultralytics", "wandb", "pydicom"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up VinBigData Chest X-ray project environment...")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing Python packages...")
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    if not verify_installation():
        print("❌ Some packages are missing")
        sys.exit(1)
    
    # Check CUDA
    print("\n🖥️ Checking CUDA availability...")
    check_cuda_availability()
    
    # Setup git hooks
    print("\n🔧 Setting up development tools...")
    setup_git_hooks()
    
    print("\n" + "=" * 60)
    print("✅ Environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Download the VinBigData dataset from Kaggle")
    print("2. Run data preparation: python main.py data --data_dir /path/to/dataset")
    print("3. Start training: python main.py train_classification --data_dir /path/to/processed/data")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
