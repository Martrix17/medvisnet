# ViT COVID-19 Classification

Vision Transformer-based classification pipeline for COVID-19 chest X-rays.  
Built with **PyTorch**, **Hydra**, **MLflow** tracking, and **ONNX** export.

## Features

- 🔬 **Multi-class classification**: COVID-19, Normal, Viral Pneumonia, Lung Opacity
- 🎯 **Vision Transformer models**: Pre-trained ViT architectures from torchvision
- ⚙️ **Hydra configuration**: Hierarchical YAML configs with experiment management
- 📊 **MLflow tracking**: Automatic logging of metrics, hyperparameters, and artifacts
- 💾 **Checkpointing**: Patience-based saving with training resumption
- 🚀 **Multiple inference backends**: PyTorch (.pt) and ONNX Runtime (Python and C++)
- ✅ **DevContainer ready**: VS Code development environment included

## Quick Start

### Installation

**Option 1: Local Installation (Python only)**

```bash
# Clone repository
git clone https://github.com/Martrix17/medvisnet.git
cd medvisnet

# Install Python dependencies
pip install -r requirements.txt
```

**Option 2: DevContainer (Recommended for C++ inference)**

VS Code with Docker installed:
1. Open project in VS Code
2. Click "Reopen in Container"

Container uses a PyTorch base image with conda environment and includes Python dependencies and C++ build tools (CMake, OpenCV, ONNX Runtime).

### Training

```bash
# Train with default configuration
python src/train.py

# Train with experiment preset
python src/train.py experiment=vit_freeze

# Override specific parameters
python src/train.py trainer.epochs=50 model.freeze_backbone=true
```

### Testing

```bash
# Test with default checkpoint
python src/test.py

# Test specific model
python src/test.py trainer.checkpoint.filename="other_model.pt"
```

### Inference

#### **Python (PyTorch or ONNX)**:

```bash
# PyTorch backend
python src/infer.py

# ONNX backend
python src/infer.py inference.backend="onnx" inference.image_path="path/to/image.png"
```

#### **C++ (ONNX only)**:

```bash
cd inference_cpp
mkdir build && cd build
cmake ..
make
./inference ../models/exported/model.onnx path/to/image.png
```

## Dataset

**Source**: [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

**Classes**:
- COVID-19 positive cases
- Normal (healthy) cases
- Viral Pneumonia cases
- Lung Opacity cases
 
Download and place extracted dataset in `data/` directory:

 ```bash
data/COVID-19_Radiography_Dataset/
├── COVID/images/
├── Normal/images/
├── Viral Pneumonia/images/
└── Lung_Opacity/images/
 ``` 

## Configuration

The project uses Hydra for hierarchical configuration with priority order:

1. `config/train.yaml` - Base training defaults
2. `config/experiment/*.yaml` - Experiment-specific overrides
3. Command-line arguments - Highest priority

### Create new experiment:

1. Create new experiment file: config/experiment/new_experiment.yaml
2. Override desired parameters
3. Run experiment training with:

```bash
python src/train.py experiment=new_experiment
```
### Key configuration groups:

- `config/data/` - Dataset parameters
- `config/model/` - Model architectures
- `config/trainer/` - Training settings
- `config/logging/` - Logger configuration

All run configurations are saved with time markers to `data/YYYY-MM-DD/HH-MM-SS/` directory.

## Checkpointing & Export

**Checkpoints** are saved automatically when validation loss improves:

```bash
models/checkpoints/base_model.pt
```

**Resume training** from checkpoint:

```bash
python src/train.py trainer.load_checkpoint=true
```

**ONNX export** happens automatically after training with models saved to:

```bash
models/exported/{run_name}.onnx
```

Models are named using the logger `run_name` (configured in `config/logging/`). Restarting a run without changing the `run_name` will overwrite the previous checkpoint.

## MLflow Tracking

Automatically logged metrics and artifacts:

- **Training/Validation**: Losses and learning rate
- **Validation**: Accuracy, precision, recall, F1, AUROC
- **Testing**: Confusion matrix, AUROC, ROC curves, classification report artifacts

**View experiments**: 

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

## 📁 Project Structure

```markdown
.
├── config/                      # Hydra YAML configurations
│   ├── data/                    # Dataset configs
│   ├── experiment/              # Experiment presets
│   ├── logging/                 # MLflow settings
│   ├── model/                   # Model architecture
│   ├── trainer/                 # Train/test settings
│   ├── train.yaml               # Main training config
│   ├── test.yaml                # Main testing config
│   └── infer.yaml               # Main inference config
│
├── src/
│   ├── data/                    # Dataset and dataloaders
│   ├── inference/               # Python inference backends
│   ├── models/                  # Vision Transformer wrapper
│   ├── training/                # Training pipeline
│   └── utils/                   # Helpers and utilities
│
├── inference_cpp/               # C++ ONNX inference
│   ├── include/inferencer.h
│   ├── src/inferencer.cpp
│   ├── src/main.cpp
│   └── CMakeLists.txt
│
├── tests/                       # Unit tests
├── .devcontainer/               # VS Code dev container
├── models/                      # Checkpoints and exports
├── mlruns/                      # MLflow tracking
├── outputs/                     # Hydra run outputs
├── results/                     # Test/inference results
│
├── train.py                     # Training entry point
├── test.py                      # Testing entry point
├── infer.py                     # Inference entry point
└── requirements.txt             # Python dependencies
```

## Development

Run all tests with coverage:

```bash
# All tests with coverage
pytest -v --cov=src --cov-report=term-missing

# Quick run (stop on first failure)
pytest -v --maxfail=1 --disable-warnings
```

**Pre-commit hooks** (black, isort, flake8):

```bash
pre-commit install
pre-commit run --all-files
```

**CI/CD**: GitHub Actions runs linting and tests on every push to `main`.

## Results

The results for each model use the same default configurations from `config/`.

| **Model** | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
|:---------:|:------------:|:-------------:|:----------:|:------------:|
| ViT-B/16  |     0.95     |     0.95      |    0.96    |    0.96      | 
| ViT-B/32  |     0.93     |     0.93      |    0.94    |    0.94      | 

See the `results/testing/` directory for more details (per-class metrics, images, predictions).

## Citation

- M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676. [Paper link](https://ieeexplore.ieee.org/document/9144185)

- Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. [Paper Link](https://www.sciencedirect.com/science/article/pii/S001048252100113X?via%3Dihub)

## Acknowledgments

- **Dataset**: COVID-19 Radiography Database by Rahman et al.
- **Models**: Vision Transformer implementations from torchvision
- **Framework**: PyTorch, Hydra, MLflow communities
