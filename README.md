# SATC Outfit Predictor

A multimodal machine learning project that predicts **which *Sex and the City* character**—Carrie, Miranda, Charlotte, or Samantha—would most likely wear a given outfit, based **purely on clothing style**, not identity cues.

This project explores whether *stylistic signals* (silhouette, color palette, formality, aesthetic) can be learned from images and text using modern embedding models.

---

## Project Goal

The core research question is:

> Can we distinguish character *style* independently of the person wearing the clothes?

To answer this, the pipeline aggressively removes identity information (faces, hair, backgrounds) and evaluates multiple classifiers over CLIP- and MPNet-based embeddings.

---

## Approach Overview

### 1. Data Collection
- ~20–60 outfit images per character
- Sources: fashion blogs, curated image search, fan archives
- Labels: `{carrie, miranda, charlotte, samantha}`

---

### 2. Identity Suppression & Preprocessing
To prevent the model from cheating via facial or contextual cues:

- Background removal using **U²-Net** (`rembg`)
- Face and hair blackout using **OpenCV DNN face detection**
- Gamma correction + CLAHE for lighting normalization
- Aspect-ratio–preserving resize with padding
- Color normalization

This ensures the model focuses on **clothing features only**.

---

### 3. Multimodal Embeddings

#### Image Embeddings
- **CLIP ViT-B/32**
- L2-normalized image embeddings

#### Text Embeddings (Outfits)
- CLIP text embeddings derived from structured metadata:
  - short description
  - aesthetic/style
  - formality level

#### Text Embeddings (Characters)
- **MPNet (all-mpnet-base-v2)**
- Long-form style descriptions per character
- Encodes high-level stylistic archetypes

---

### 4. Models Evaluated

| Model | Description |
|------|------------|
| **clip_fusion_linear** | Multinomial, class-balanced logistic regression over concatenated CLIP image + text embeddings |
| **clip_fusion_ablation** | Same as Model A, evaluated under modality ablations (image-only and text-only inputs) |
| **style_space_linear** | Projection-based comparison between outfit embeddings and character-style embeddings |
| **style_space_mlp** | Shallow MLP over concatenated embeddings |

---

### 5. Evaluation

- **Stratified K-fold cross-validation**
- Metrics:
  - Accuracy
  - Macro Precision / Recall / F1
  - Per-class metrics

**Typical performance:**
- Accuracy: ~0.59–0.62  
- Macro F1: ~0.59–0.61

Charlotte had the highest proportion of correctly classified outfits, while Miranda had the lowest.

---

## Example Metrics

```text
====================================
CROSS-VALIDATION SUMMARY
====================================
Accuracy:        0.595 ± 0.048
Macro Precision: 0.598 ± 0.048
Macro Recall:    0.596 ± 0.054
Macro F1:        0.589 ± 0.052

====================================
     PER-CLASS METRICS (AGGREGATED)
====================================
carrie      Precision: 0.6950 ± 0.1075   Recall: 0.6712 ± 0.1029   F1: 0.6728 ± 0.0600
charlotte   Precision: 0.7326 ± 0.1090   Recall: 0.6655 ± 0.0777   F1: 0.6933 ± 0.0756
miranda     Precision: 0.6444 ± 0.3014   Recall: 0.4679 ± 0.2168   F1: 0.5258 ± 0.2295
samantha    Precision: 0.5970 ± 0.0445   Recall: 0.7379 ± 0.1206   F1: 0.6584 ± 0.0747
```

## Repository Structure

satc-outfit-predictor/
├── data/
│   ├── raw_images/
│   │   ├── carrie/
│   │   ├── charlotte/
│   │   ├── miranda/
│   │   └── samantha/
│   ├── processed_images/
│   │   ├── carrie/
│   │   ├── charlotte/
│   │   ├── miranda/
│   │   └── samantha/
│   └── metadata/
│       ├── all_outfits.csv
│       └── style_descriptions.txt
│
├── src/
│   ├── preprocessing/
│   │   ├── remove_background.py
│   │   ├── blackout_faces.py
│   │   └── normalize_size_lighting.py
│   └── models/
│       ├── clip_fusion_linear.py
│       ├── clip_fusion_ablation.py
│       ├── style_space_linear.py
│       └── style_space_mlp.py
│
├── assets/
│   └── face_detector/
│       ├── deploy.prototxt
│       └── res10_300x300_ssd_iter_140000.caffemodel
│
├── requirements.txt
└── README.md

---

## Takeaway

This project demonstrates that character style is partially learnable from clothing alone, while also exposing the ambiguity and subjectivity of fashion classification. Image-based embeddings outperformed text-only embeddings by roughly 10% in an ablation study, leaving open the question of how much this advantage reflects true visual style versus subtle residual bias in the images—suggesting that additional preprocessing may be required.

---

## License

MIT License 

---

