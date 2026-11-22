# License Plate Character Recognition (Persian & English)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-YOLOv8m-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning-based computer vision project for detecting and recognizing characters inside vehicle license plates using **YOLOv8m**, supporting both **Persian** and **English** character recognition.

## Dataset
This project utilizes the **Large Dataset of Persian License Plate Characters** available on Kaggle:
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/amirebrahimi66/large-dataset-of-persian-license-plate-characters)

## Demo

![Sample Output](assets/8.jpg)

## Features
- Accurate license plate detection and character recognition using a pre-trained **YOLOv8m** model.
- Dual-language support (`recognize_fa.py` for Persian, `recognize_en.py` for English).
- Lightweight repository with pre-trained weights hosted via GitHub Releases.

## Project Structure
```text
├── assets/            # Demo output images
├── models/            # Directory for pre-trained .pt model weights
├── samples/           # Sample test images
├── recognize_fa.py    # Persian character recognition script
├── recognize_en.py    # English character recognition script
├── requirements.txt   # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
