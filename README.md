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

## How to Run
```bash
pip install torch torchvision datasets pillow opencv-python numpy
python train.py
python app.py
