# AI-Generated vs Real Image Classification

## Overview
This project builds a **PyTorch Convolutional Neural Network (CNN)** to classify images as **AI-generated** or **real**.  
The model is trained on a large-scale Hugging Face dataset and achieves **~90% accuracy**.

To improve interpretability, the project uses **Grad-CAM** to visualize which regions of an image most influenced the model’s prediction.  
A simple **Tkinter GUI** allows users to upload images and view predictions and heatmaps interactively.

---

## Dataset
- **Source:** Hugging Face  
- **Name:** `Hemg/AI-Generated-vs-Real-Images-Datasets`
- **Size:** ~150,000 images  
- **Classes:**  
  - `0` — Real Image  
  - `1` — AI-Generated Image  

---

## Model
- Custom CNN built from scratch in PyTorch
- 3 convolutional layers with ReLU and max pooling
- Fully connected layers for binary classification
- **Loss:** CrossEntropyLoss  
- **Optimizer:** Adam  

---

## Explainability (Grad-CAM)
Grad-CAM highlights image regions that most affected the model’s decision:
- 🔴 Red → High importance
- 🔵 Blue → Low importance

This helps analyze model behavior and detect reliance on visual artifacts.

---

## GUI Demo
The Tkinter interface allows users to:
1. Upload an image
2. View the predicted label and confidence
3. Display a Grad-CAM heatmap overlay

---

## Results
- **Accuracy:** ~90%
- **Task:** Binary image classification (AI vs Real)

---

## Tech Stack
- Python
- PyTorch
- Torchvision
- Hugging Face Datasets
- OpenCV
- PIL
- Tkinter
- NumPy

---
## Limitations

Despite achieving strong performance (~90% accuracy), the model has several limitations:

- **Sensitivity to High-Quality AI Images:** Modern AI-generated images can be highly realistic and lack obvious artifacts. In these cases, the model may misclassify AI images as real because it relies on global texture consistency rather than explicit synthetic indicators.

- **Dataset Bias and Generalization:** The model is trained on a specific dataset and may learn dataset-dependent patterns. Performance can degrade when evaluated on images from unseen sources or newer generative models.

- **Limited Grad-CAM Interpretability:** Grad-CAM highlights regions that influence predictions but does not explain the underlying reasoning. Activations are often distributed across the image, making it difficult to isolate specific features driving the decision.

- **Sensitivity to Preprocessing Changes:** Accuracy may fluctuate when images are converted to grayscale or undergo distribution shifts, indicating partial reliance on color-based cues.

- **Binary Classification Scope:** The model only distinguishes between AI-generated and real images and does not identify the source model, detect partial manipulations, or assess confidence in authenticity beyond a single score.

- **Confidence vs. Correctness:** High confidence predictions do not always correspond to correct classifications, particularly for visually ambiguous images. The model should be used as a decision-support tool rather than a definitive detector.

---
## How to Run
```bash
python train.py
python app.py
