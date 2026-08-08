# My Methodology

## 1. Environment
* **OS**: Ubuntu  24.04.2 LTS
* **Python**: 3.12.3
* **PyTorch**: 2.9.1+cu128

## 2. Installation
### 2.1 Clone this project
```bash
cd ~
git clone https://github.com/Coding-God-Vincent/GenDAC.git
cd GenDAC
```

### 2.2 Create new conda environment
```bash
# Create environment with Python 3.12.3
conda create -n GenDAC_venv python=3.12.3 -y

# Activate environment
conda activate GenDAC_venv

# Install Pytorch 2.9.1 with CUDA 12.8
conda install pytorch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r reqirements.txt
```

## 3. Repository structure
