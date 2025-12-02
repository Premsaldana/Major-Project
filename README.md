# NeuroSpine AI 

**A Clinical-Grade Deep Learning System for Spine X-Ray Diagnosis & Reporting**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Ensemble-EE4C2C?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

##  Overview

**NeuroSpine AI** is an advanced medical imaging web application designed to assist radiologists in the detection and classification of spinal abnormalities. It utilizes an **Ensemble Deep Learning Model** combining **ConvNeXt Tiny** and **EfficientNet-B3** to achieve state-of-the-art accuracy on lumbar spine X-rays.

Beyond simple classification, the system provides **Explainable AI (XAI)** visuals using **Grad-CAM++** to generate high-resolution heatmaps that pinpoint the exact location of injuries. It also features an automated **Radiology Report Generator** that creates professional PDF reports with clinical recommendations.

##  Key Features

* ** High-Accuracy Ensemble:** Uses soft-voting ensemble learning with Test-Time Augmentation (TTA) to combine predictions from ConvNeXt and EfficientNet architectures.
* ** Clinical Explainability:** Generates high-resolution **Grad-CAM++ heatmaps** overlaid on the original high-res X-ray to visually verify the model's focus.
* ** Advanced Preprocessing:** Implements **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance bone trabecular structure and edge visibility in low-contrast scans.
* ** Automated Reporting:** Instantly generates a downloadable **PDF Radiology Report** following RSNA standards, complete with patient info, findings, impression, graphs, and heatmaps.
* ** Professional Interface:** A "Cockpit-style" single-page dashboard designed for efficiency, featuring a drag-and-drop diagnostic workflow.

##  Tech Stack

* **Deep Learning:** PyTorch, Torchvision (ConvNeXt, EfficientNet)
* **Backend:** Flask (Python)
* **Image Processing:** OpenCV, NumPy, Pillow, Matplotlib
* **Frontend:** HTML5, CSS3 (Grid Layout), JavaScript (ES6)
* **Reporting:** jsPDF, Chart.js logic

##  Project Structure

```bash
NeuroSpine-AI/
├── app.py                 # Main Flask Application & Model Inference Logic
├── best_model_convnext.pth # Trained Weights for ConvNeXt Model
├── best_model_effnet.pth   # Trained Weights for EfficientNet Model
├── static/
│   ├── styles.css         # Professional Dashboard Styling
│   └── script.js          # Frontend Logic & PDF Report Generation
├── templates/
│   └── index.html         # Main Web Interface
└── README.md              # Documentation
