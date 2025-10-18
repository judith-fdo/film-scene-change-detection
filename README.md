# Scene Change Detection in Film Production 🎬

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Hybrid deep learning pipeline for automatically detecting added, removed, and position-changed objects in film scenes.**

**Competition Result:** 🎖️ **4th Place** | **Kaggle F1:** 0.523 | **Validation F1:** 0.574

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Key Learnings](#key-learnings)
- [Team](#team)

---

## 🎯 Overview

Automated detection of object-level changes (added, removed, position-changed) between paired film scene images using a novel 5-stage hybrid deep learning pipeline.

**Key Achievement:** Combined Siamese Networks, YOLOv11, CLIP, and Hungarian Algorithm to achieve competitive performance with limited resources (6GB GPU).

---

## 🎬 Problem Statement

**Input:** Two images from the same scene (Before & After)  
**Output:** Three categories of changes:

- ✅ **Added Objects** - New objects appearing in second image
- ✅ **Removed Objects** - Objects disappeared from first image
- ✅ **Changed Objects** - Objects with significant position/appearance changes

### Challenges

- Objects vary dramatically in size (small props to large set pieces)
- Complex, cluttered scenes with multiple overlapping objects
- Must distinguish object movement from camera movement
- Multi-label prediction (3 simultaneous categories)
- Class imbalance (20% of image pairs have no changes)

---

## 🏗️ Solution Architecture

**5-Stage Hybrid Pipeline:**
```
1. Siamese Network (ResNet-18)
   ↓ Determines IF changes exist
   
2. YOLOv11x Object Detection
   ↓ Identifies WHAT objects present
   
3. CLIP Feature Extraction
   ↓ Generates semantic embeddings
   
4. Hungarian Algorithm
   ↓ Optimal object matching
   
5. Change Classification
   ↓ Categorizes as Added/Removed/Changed
```

**Why Hybrid?**
- Modular design enables independent optimization
- Leverages state-of-the-art pre-trained models
- Interpretable with clear failure points
- Efficient pre-screening saves computation

---

## 📊 Results

### Competition Performance

| Submission | Configuration | Added F1 | Removed F1 | Changed F1 | Overall F1 (Kaggle) | Δ |
|------------|---------------|----------|------------|------------|---------------------|---|
| 1 | Baseline (Siamese + YOLOv8) | ~0.48 | ~0.44 | ~0.40 | 0.5015 | - |
| 2 | + Hungarian + Optimization | ~0.49 | ~0.50 | ~0.55 | 0.5130 | +0.0115 |
| 3 | + YOLOv11 | ~0.51 | ~0.57 | ~0.64 | **0.5230** | +0.0100 |

**Note:** Category F1 scores are validation estimates (~ indicates approximate values)
**Validation-Test Discrepancy:** We achieved 0.574 F1 on our validation set but 0.523 on the Kaggle leaderboard—a gap of 0.051 that emphasizes the critical importance of representative validation splits.

### Leaderboard

| Rank | Score |
|------|-------|
| 🥇 1st | 0.5946 |
| 🥈 2nd | 0.5866 |
| 🥉 3rd | 0.5732 |
| **🎖️ 4th** | **0.5230** |

### Novel Discoveries

🔍 **YOLOv11 alone > Ensemble**
- YOLOv11: 0.574 F1
- YOLOv8+v11: 0.552 F1
- Lesson: Quality > Quantity

🔍 **Hungarian > Greedy**
- Hungarian: 0.574 F1
- Greedy: 0.553 F1
- Gain: +0.021 F1 (3.8%)

### Ablation Study

| Component Removed | F1 | Impact |
|-------------------|-----|--------|
| (Full Pipeline) | 0.574 | - |
| Siamese Network | 0.501 | -0.073 🔴 |
| CLIP Features | 0.512 | -0.062 🔴 |
| Hungarian | 0.553 | -0.021 🟡 |
| Post-processing | 0.569 | -0.005 🟢 |

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- CUDA 12.1+ (for GPU)
- 6GB+ GPU memory

### Setup
```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/scene-change-detection.git
cd scene-change-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Dataset not included due to size. Required structure:
```
Data/
├── train.csv
├── test.csv
└── data/
    └── data/
        ├── {id}_1.png
        └── {id}_2.png
```

---

## 💻 Usage

### Run Main Notebook
```bash
# Activate environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Launch Jupyter
jupyter notebook

# Open: Notebooks/try02.ipynb
```

### Notebook Sections

1. **Data Loading & EDA** - Explore dataset distribution
2. **Siamese Network Training** - Train change detector
3. **Pipeline Implementation** - Full 5-stage system
4. **Evaluation** - Validation metrics
5. **Test Predictions** - Generate submissions

### Quick Test
```python
# In notebook or Python script
from pathlib import Path
import torch

# Load data
data_path = Path('../Data')
train_df = pd.read_csv(data_path / 'train.csv')

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
siamese = torch.load(data_path / 'best_siamese_model.pt')
```

---

## 📁 Project Structure
```
Octwave Final/
│
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
│
├── Data/
│   ├── train.csv               # Training labels (4,536 pairs)
│   ├── test.csv                # Test IDs (1,482 pairs)
│   ├── best_siamese_model.pt   # Trained weights
│   ├── submission_yolov11_final.csv  # Final submission
│   └── data/
│       └── data/               # Images (not in GitHub)
│           ├── {id}_1.png     # Before images
│           └── {id}_2.png     # After images
│
├── Notebooks/
│   ├── try02.ipynb            # Main pipeline (ALL CODE HERE)
│   └── (other experiments)     # Exploration notebooks
│
└── venv/                       # Virtual environment (excluded)
```

**Note:** Images and models excluded from GitHub due to size.

---

## 🔧 Technical Details

### Component 1: Siamese Network

**Purpose:** Fast change detection pre-screening

- **Architecture:** ResNet-18 with shared weights
- **Input:** 224×224 RGB (both images)
- **Output:** 3 binary predictions
- **Training:**
  - Loss: Binary Cross-Entropy
  - Optimizer: AdamW (lr=0.001)
  - Epochs: 15 (early stop at 10)
  - Val Loss: 0.5897
  - Time: ~2.5h on RTX 4050

**Why?** Designed for comparison, efficient, transfer learning

### Component 2: YOLOv11x

**Purpose:** Object detection

- **Config:** Confidence 0.22, Input 640×640
- **Pre-trained:** COCO (80 classes)
- **Why v11?** Superior to v8 and ensemble (+0.021 F1)

### Component 3: CLIP ViT-B/32

**Purpose:** Semantic feature extraction

- **Features:** 512-D embeddings
- **Similarity:** Cosine (threshold 0.65)
- **Why?** Robust to transformations, -0.062 F1 when removed

### Component 4: Hungarian Algorithm

**Purpose:** Optimal object matching

- **Complexity:** O(n³), fast for 3-8 objects
- **Why?** Globally optimal, +0.021 F1 over greedy

### Component 5: Classification

**Logic:**
- Matched + (Δpos > 45px OR Δsize > 25%) → Changed
- Unmatched in img1 → Removed
- Unmatched in img2 → Added

### Optimized Hyperparameters

| Parameter | Value | Method |
|-----------|-------|--------|
| Siamese Threshold | 0.45 | Grid Search |
| YOLO Confidence | 0.22 | Grid Search |
| CLIP Similarity | 0.65 | Grid Search |
| Position Threshold | 45px | Grid Search |
| Size Change | 0.25 | Grid Search |

---

## 🎓 Key Learnings

### Technical

✅ Transfer learning powerful (ImageNet, COCO, CLIP)  
✅ Optimal algorithms matter (Hungarian > Greedy)  
✅ Model quality > quantity (v11 alone > ensemble)  
✅ Systematic tuning essential (+0.02-0.03 F1)  
✅ Ablation reveals importance (Siamese & CLIP critical)

### Practical

✅ Validation can mislead (0.051 gap: val 0.574 → test 0.523)  
✅ Test early on leaderboard, don't trust val alone  
✅ Modular design enables iteration  
✅ Constraints force creativity (6GB GPU → efficient choices)

### Challenges Overcome

- Limited GPU (6GB) → Batch size 32, ResNet-18
- Windows multiprocessing → num_workers=0
- Val-test gap → Representative splits crucial
- Model selection → Systematic testing

---

## 🚀 Future Improvements

### Short-term (+0.03-0.05 F1)

- [ ] Adaptive thresholds (scale with object size)
- [ ] Multi-scale YOLO (640 + 1280px)
- [ ] Camera motion compensation
- [ ] Confidence calibration

### Long-term (+0.10-0.15 F1)

- [ ] Fine-tune YOLO on film scenes
- [ ] End-to-end trainable architecture
- [ ] Attention mechanisms
- [ ] Temporal modeling (multi-frame)

---

## 👥 Team

**Team Octwave**

- **[Your Name]** - [Role]
- **[Name 2]** - [Role]
- **[Name 3]** - [Role]

---

## 🙏 Acknowledgments

- Competition organizers
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv11
- [OpenAI](https://github.com/openai/CLIP) - CLIP
- [PyTorch](https://pytorch.org/) team
- Open-source ML community

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 📞 Contact

- **GitHub:** [YOUR-USERNAME]
- **Email:** [your.email@example.com]

---

**⭐ If you found this helpful, please star the repo!**

---

*Built with ❤️ by Team Octwave*
```

