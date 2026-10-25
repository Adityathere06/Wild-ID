
# 🐅 WildID — AI-Powered Individual Tiger Identification System

**Using AI and Computer Vision to Support Wildlife Conservation**

WildID is an AI-powered tiger re-identification prototype designed to distinguish individual tigers by comparing their unique stripe patterns with images stored in a reference database.

Developed for **WildHack 2026**, this project explores how computer vision and deep learning can support individual wildlife identification and potentially assist wildlife monitoring and conservation efforts.

## 🌿 Problem Statement

Identifying individual tigers is important for wildlife monitoring, population studies, and conservation research. However, identifying the same tiger across different images can be challenging because of variations in viewing angles, poses, lighting conditions, and image quality.

WildID addresses this challenge by extracting visual feature embeddings from tiger images and comparing them against stored reference images to find potential matches.

## 🚀 Key Features

### 1. Individual Tiger Re-Identification
- Compares a query image with reference images in the database.
- Ranks potential matches using visual feature similarity.
- Supports identifying known individuals and flagging potential new individuals.

### 2. Multi-Angle Identification
The system is designed to compare tiger images captured from different viewing angles and poses.

Because the visible appearance of stripe patterns can change with perspective, multi-angle matching is an important challenge in individual tiger re-identification.

### 3. Lighting Variation Considerations
Wildlife images may be captured under different brightness levels, shadows, and environmental conditions.

WildID explores image matching under these variations, with robustness depending on image quality, model performance, and further testing.

### 4. Webcam-Based Identification
- Captures images through a webcam.
- Processes captured frames for tiger identification.
- Displays potential matches and similarity scores.
- Supports experimentation with images displayed on another screen.

### 5. Batch Identification
Processes multiple test images in one run and reports potential matches against the reference database.

### 6. Reference Database
Stores reference images associated with individual tiger IDs, allowing the system to compare new images without requiring a new model-training run whenever a new individual is added.

## 🧠 Technical Approach

WildID uses a two-stage identification pipeline:

1. **Species Filtering:** A MobileNetV3-based model checks whether the image contains a tiger.
2. **Feature Extraction:** A fine-tuned Vision Transformer (ViT-Base) generates a visual embedding.
3. **Embedding Projection:** A neural network maps the 768-dimensional ViT representation to a normalized 128-dimensional embedding.
4. **Similarity Matching:** Cosine similarity compares the query embedding with reference embeddings.
5. **Identity Decision:** The system ranks candidate individuals and applies a similarity threshold to determine whether a potential match is found.

### Pipeline

```text
Input Tiger Image
       |
Image Preprocessing
       |
MobileNetV3 Species Filter
       |
Fine-Tuned ViT-Base
       |
128-Dimensional Feature Embedding
       |
Cosine Similarity
       |
Reference Tiger Database
       |
Potential Individual Match / Unknown
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Species Filtering | MobileNetV3 |
| Feature Extraction | Vision Transformer (ViT-Base) |
| Model Training | PyTorch |
| Training Objective | Triplet Loss |
| Similarity Metric | Cosine Similarity |
| Image Processing | Image preprocessing and normalization |
| Interactive Interface | Gradio |
| Version Control | Git and GitHub |

*The stack above reflects the components documented for this project. Update it if the final implementation differs.*

## 📊 Model Training Summary

| Parameter | Value |
|---|---|
| Total individual tigers in training dataset | 107 |
| Training split | 74 tigers |
| Holdout/reference split | 33 tigers |
| Random seed | 42 |
| Epochs trained | 19 |
| Best recorded triplet loss | 0.0054 (Epoch 9) |
| Embedding dimension | 128 |
| Cosine similarity threshold | 0.97 |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Learning rate scheduler | CosineAnnealingLR |

## 🌍 Future Scope

Our vision is to extend WildID beyond tigers and explore individual re-identification for other wildlife species.

Potential future improvements include:

- Improving robustness across different angles, poses, and lighting conditions.
- Expanding the reference database with diverse images of known individuals.
- Evaluating performance with precision, recall, rank-based identification metrics, and false-match rates.
- Integrating wildlife camera-trap data and metadata.
- Exploring support for other species based on their distinctive visual characteristics.
- Evaluating the system in collaboration with wildlife researchers and conservation teams.

## 🎯 Project Objective

WildID demonstrates how AI and computer vision can be applied to the challenge of individual tiger re-identification. The goal is to develop a more robust and scalable approach that may support wildlife research and conservation in the future.

---

**Built with Python, deep learning, and a goal of supporting wildlife conservation. 🐅🌿**

---

## Project Structure

```

minor/
├── data/                        # Download separately (see below)
│   ├── fine_tuned_tigers/       # 74 training tigers
│   ├── reference/               # 33 holdout tigers (reference DB)
│   └── test/                    # Query images
├── models/
│   ├── tiger_finetuned_vit.pth  # Download separately (see below)
│   └── reference_features.pkl   # Pre-built reference embeddings
├── src/
│   ├── feature_extractor.py
│   ├── matcher.py
│   └── utils.py
├── results/                     # Runtime outputs saved here
├── demo.py                      # Main Gradio UI
├── model.py                     # ViT + embedding head definition
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Adityathere06/Wild-ID.git
cd Wild-ID
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Download Required Files

### 1. Model Weights (329MB)
Download and place inside `models/`:

**[Download tiger_finetuned_vit.pth](https://drive.google.com/file/d/1GZ8hb2vX3JBfK89WqR7KZZsOHSaAdXuc/view?usp=sharing)**

### 2. Dataset
Download the zip, extract it, and place the `data/` folder in the project root:

**[Download data.zip](https://drive.google.com/file/d/1eWLyGOAd_Lt4wO07_dwzRsTHuahO5Af3/view?usp=sharing)**

```
data/
├── fine_tuned_tigers/
├── reference/
└── test/
```


---


## Run

```bash
python demo.py
```

Opens Gradio UI at `http://127.0.0.1:7861`

---

## Tech Stack

- Python 3.x
- PyTorch + torchvision
- timm (ViT-Base pretrained on ImageNet-21k)
- Gradio
- scikit-learn
- albumentations
README
```
