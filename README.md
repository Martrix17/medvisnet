# ViT COVID-19 Classification

Vision Transformer-based classification pipeline for COVID-19 chest X-rays.  
Built with **PyTorch**, **Hydra**, **MLflow**, and a modular trainer framework.

## 📊 Dataset

Contains chest X-ray images across multiple classes:

- COVID-19 positive cases
- Normal (healthy) cases
- Viral Pneumonia cases
- Lung Opacity cases

Source: [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

## 🚀 Features

- **DevContainer** configuration with **PyTorch** base image
- Modular **training architecture** (`BaseTrainer`, `Trainer`)
- **Hydra-based configuration** for **experiments configurations** via YAML files
- **MLflow logging** for metrics, hyperparameters, and artifacts
- **Checkpointing** and **early stopping**
- **Test coverage** with CI/CD

## 🧠 Running Training

Run an experiment directly:

```python src/train.py experiment=vit_freeze```

Or override configs dynamically:

```python src/train.py trainer.epochs=20 model.model.freeze_backbone=false```

## 🧪 Running Testing

Testing will load the file specified:

```python src/test.py```

## Running Inference

_To be added: C++ and Python inference examples._

## 📁 Project Structure

```markdown
.
├── .devcontainer/               # VS Code dev container config
│   ├── devcontainer.json
│   └── Dockerfile
│
├── config/                      # Hydra configuration files
│   ├── data/                    # Dataset configs
│   ├── experiment/              # Experiment presets
│   ├── logging/                 # MLflow settings
│   ├── model/                   # Model architecture config
│   ├── trainer/                 # Train/test trainer configuration
│   ├── train.yaml               # Main training config
│   └── test.yaml                # Main testing config
├── outputs/                     # Hydra outputs
│
├── experiments/
│   ├── checkpoints/             # Saved checkpoints as .pt files
│   └── onnx/                    # Saved models in onnx format
|
├── src/
│   ├── data/                    # Data loading and preprocessing
│   │   ├── dataloader.py
│   │   └── dataset.py
│   ├── models/
│   │   └── vit.py               # Vision Transformer implementation
│   ├── training/
│   │   ├── base_trainer.py      # Abstract base trainer
│   │   ├── trainer.py           # Main training loop
│   │   ├── callbacks.py         # Early stopping and callbacks
│   │   ├── factories.py         # Object factories
│   │   └── metrics.py           # Evaluation metrics
│   └── utils/
│       ├── checkpoint.py        # Checkpoint utilities
│       ├── logger.py            # Logging utilities
│       └── helper.py            # Helper functions
│
├── tests/                       # Unit and integration tests
├── inference_cpp/               # C++ inference implementation
│   └── CMakeLists.txt
│
├── train.py                     # Training entry point
├── test.py                      # Testing entry point
├── results                      # Locally saved outputs
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata
├── setup.py                     # Package setup
├── Makefile                     # Common commands
├── .flake8                      # Flake8 configuration
├── .pre-commit-config.yaml      # Pre-commit hooks
└── README.md                    # This file
```

## ⚙️ Configuration

The project uses Hydra for hierarchical configuration management.

Configuration hierarchy:

1. ```config/train.yaml``` → global training defaults
2. ```config/experiment/*.yaml``` → experiment-specific overrides
3. Command-line arguments → highest priority

To create a new experiment:

1. Add a YAML file in config/experiment/
2. Override any parameters
3. Run: ```python src/train.py experiment=new_experiment_name```

## 📊 MLflow Logging

Metrics and artifacts are automatically logged:

- Training and validation losses
- Accuracy, F1, precision, recall, AUROC
- Confusion matrix and ROC curves

Start MLflow UI: ```mlflow ui --backend-store-uri mlruns```

## 💾 Checkpoints

Models (by validation loss) are saved automatically during training:

```experiments/checkpoints/base_model.pt```

Resume training:

```python src/train.py load_checkpoint=true resume_training=true```

## 🧪 Testing

Run all tests with coverage:

```pytest -v --cov=src --cov-report=term-missing```

To see which tests ran:

```pytest -v --maxfail=1 --disable-warnings```

## ⚙️ Continuous Integration

GitHub Actions workflow runs on every push or PR to main:

- Linting (black, isort, flake8)
- Unit tests with coverage
- Coverage summary appended to workflow logs

File: ```.github/workflows/ci.yml```

## 📈 Results

_To be added: model performance metrics and plots._

## Model Export

_To be added: ONNX / TorchScript export steps._

## 📝 Citation

- M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676. Paper link

- Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. Paper Link
To view images please check image folders and references of each image are provided in the metadata.xlsx.

## 🙌 Acknowledgments

Dataset
COVID-19 Radiography Database by Tawsifur Rahman et al.
