# Evaluating the Effectiveness of Posterior Truncation in Mitigating Membership Inference Attacks

This repository contains the implementation and experimental framework for evaluating posterior truncation as a defense against membership inference attacks on public image classification models.

## 📋 Overview

Membership inference attacks (MIAs) pose significant privacy risks by determining whether specific data points were used to train machine learning models. This work provides the first comprehensive evaluation of posterior truncation defenses on realistic pretrained models, demonstrating up to 0.232 AUC reduction in attack effectiveness with statistical significance.

### Key Findings
- **Strong overfitting-vulnerability correlation** (r=0.903 for entropy-based attacks)
- **Significant defense effectiveness** with top-1 truncation (p=0.016, Cohen's d=2.83)
- **Attack-specific patterns** - MPE attacks highly vulnerable, PCA attacks immune
- **Minimal computational overhead** (<1ms per batch for all truncation strategies)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- NVIDIA GPU (recommended, but CPU works)
- 8GB+ RAM
- 50GB+ storage for datasets and models

### Installation

```bash
# Clone the repository
git clone https://github.com/jenomgimba/posterior_truncation_project.git
cd posterior_truncation_project

# Create virtual environment
python -m venv securitynet_env

# Activate environment
# Windows:
securitynet_env\Scripts\activate
# macOS/Linux:
source securitynet_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python scripts/test_basic_models.py
```

**Expected Output:**
```
✓ ResNet-18 loaded: 11,689,512 parameters
✓ VGG-16 loaded: 138,357,544 parameters
✓ CIFAR-10 test set: 10000 samples
✓ Model inference successful
🎉 Basic setup is working!
```

## 📁 Repository Structure

```
posterior_truncation_project/
├── scripts/              # Main implementation
├── data/                 # Datasets (auto-downloaded)
├── models/               # Generated models
├── results/              # Experimental outputs
```

## 🔬 Reproduction Guide

### Step 1: Environment Verification
```bash
python scripts/test_basic_models.py
```
Verifies PyTorch installation and basic model loading.

### Step 2: Create Research Models
```bash
python scripts/create_research_models.py
```
Creates models adapted for CIFAR-10 and CIFAR-100 datasets.

### Step 3: Generate Overfitted Models
```bash
python scripts/finetune_models.py
```
**Duration:** 30-60 minutes  
**Output:** 10 models with varying overfitting levels in `models/finetuned/`

**Expected Output:**
```
✓ resnet18_cifar10_light: Overfitting = 0.143
✓ resnet18_cifar10_extreme: Overfitting = 0.305
✓ resnet18_cifar100_extreme: Overfitting = 0.685
```

### Step 4: Run Comprehensive Evaluation
```bash
python scripts/comprehensive_evaluation.py
```
**Duration:** 15-30 minutes  
**Output:** Complete experimental results and visualizations

**Expected Output:**
```
✓ Overfitting vs Attack Success plot
✓ Truncation Effectiveness Heatmap
✓ Defense Effectiveness by Model
✓ Summary Statistics Table
```

### Step 5: Statistical Analysis
```bash
python scripts/statistical_analysis.py
python scripts/error_analysis.py
python scripts/performance_analysis.py
python scripts/table_generator.py
```

## 📊 Understanding Results

### Key Output Files

| File | Description |
|------|-------------|
| `results/comprehensive_results.csv` | All experimental data (84 configurations) |
| `results/overfitting_analysis.json` | Model overfitting levels |
| `results/statistical_analysis.csv` | Significance tests and effect sizes |
| `results/figures/` | Publication-ready visualizations |
| `results/latex_tables.txt` | IEEE format tables |

### Interpreting Metrics

- **AUC (Area Under Curve)**: Attack effectiveness (0.5 = random, 1.0 = perfect)
- **Overfitting Level**: Train accuracy - Test accuracy
- **p-value**: Statistical significance (< 0.05 = significant)
- **Cohen's d**: Effect size (> 0.8 = large effect)

### Key Results

```
Top-1 Truncation vs MPE Attacks:
- CIFAR-10 Extreme: 0.732 → 0.500 AUC (-0.232, p=0.016)
- CIFAR-100 Light: 0.686 → 0.500 AUC (-0.186, p=0.016)
```

## 🔧 Configuration

### Experiment Parameters

Edit `configs/experiment_config.yaml` to customize:

```yaml
datasets:
  - name: "CIFAR-10"
    sample_size: 1000
  - name: "CIFAR-100" 
    sample_size: 1000

truncation:
  strategies: ["none", "top_k", "confidence"]
  top_k_values: [1, 3, 5]
  confidence_thresholds: [0.1, 0.3, 0.5]

evaluation:
  cross_validation_folds: 5
  num_runs: 10
```

### Hardware Optimization

**For GPU acceleration:**
```bash
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For limited resources:**
- Reduce `sample_size` in configs
- Use fewer `num_runs`
- Test subset of models first

## 📈 Extending the Work

### Adding New Attack Methods

1. Create attack class in `scripts/fixed_membership_inference_engine.py`:
```python
def your_new_attack(self, member_outputs, non_member_outputs):
    # Implement your attack logic
    return {
        'attack_method': 'YourAttack',
        'auc': calculated_auc,
        'member_scores': member_scores,
        'non_member_scores': non_member_scores
    }
```

2. Add to evaluation pipeline in `comprehensive_evaluation.py`

### Adding New Truncation Strategies

1. Extend `apply_truncation` method:
```python
elif strategy == 'your_strategy':
    # Implement truncation logic
    return truncated_posteriors
```

2. Add to configuration file

### Adding New Datasets

1. Add dataset loading in `load_datasets()` method
2. Update configuration with new dataset parameters
3. Ensure proper transforms and normalization

## 🐛 Troubleshooting

### Common Issues

**Memory Errors:**
```bash
# Reduce batch size and sample size
# Edit configs/experiment_config.yaml
```

**CUDA Errors:**
```bash
# Fall back to CPU
export CUDA_VISIBLE_DEVICES=""
```

**Slow Performance:**
```bash
# Use smaller subset for testing
python scripts/fixed_membership_inference_engine.py  # Tests 3 models only
```

**NaN Values in Results:**
```bash
# Use the fixed engine (already implemented)
python scripts/fixed_membership_inference_engine.py
```

### Platform-Specific Notes

**Windows:**
- Use `num_workers=0` in DataLoader
- Use forward slashes in paths
- Ensure PowerShell execution policy allows scripts

**macOS/Linux:**
- Standard installation should work
- May need to install build tools for some packages

### Getting Help

1. **Check logs**: All scripts print detailed progress
2. **Verify environment**: Run `test_basic_models.py` first
3. **Check GPU**: `torch.cuda.is_available()` should return `True` for GPU
4. **Validate data**: Ensure datasets downloaded correctly in `data/`


## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SecurityNet database for providing realistic pretrained models
- PyTorch and torchvision teams for deep learning framework
- National College of Ireland for computational resources

---

**For questions or issues, please open a GitHub issue or contact the author.**