Handwritten Digit Recognition using CNN
https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/TensorFlow-2.13-orange
https://img.shields.io/badge/License-MIT-yellow.svg
https://colab.research.google.com/assets/colab-badge.svg

📋 Overview
A Convolutional Neural Network (CNN) implementation for recognizing handwritten digits (0-9) using the MNIST dataset. This project demonstrates end-to-end deep learning workflow from data preprocessing to deployment.

🎯 Features
98%+ Accuracy on MNIST test set

Real-time Prediction via web interface

Multiple Input Methods: Upload images or draw directly

Comprehensive Visualization: Training history, confusion matrix, sample predictions

Production Ready: Flask web app with REST API

Modular Codebase: Clean, maintainable, and well-documented

📊 Results
Metric	Value
Test Accuracy	98.5%
Test Loss	0.045
Precision	98.6%
Recall	98.5%
F1 Score	98.5%
🏗️ Architecture
text
Input (28×28×1) → Conv2D (32 filters) → BatchNorm → MaxPooling
→ Conv2D (64 filters) → BatchNorm → MaxPooling → Flatten
→ Dense (128 units) → Dropout (0.5) → Output (10 units)
🚀 Quick Start
1. Installation
bash
# Clone the repository
git clone https://github.com/yourusername/deep-learning-portfolio.git
cd deep-learning-portfolio/projects/01_handwritten_digit_recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Train the Model
bash
# Full training
python scripts/train.py

# Quick test (small dataset)
python scripts/train.py --quick-test

# With data augmentation
python scripts/train.py --use-generator
3. Make Predictions
bash
# Predict a single image
python scripts/predict.py --image path/to/image.png

# Predict all images in a directory
python scripts/predict.py --dir path/to/images/

# Use the best model
python scripts/predict.py --model artifacts/models/best_model.h5 --image test.png
4. Run Web Application
bash
cd app
python app.py
# Open http://localhost:5000 in your browser
📁 Project Structure
text
01_handwritten_digit_recognition/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── config/
│   └── config.yaml          # Configuration
├── src/                     # Source code
│   ├── data/               # Data loading & preprocessing
│   ├── models/             # CNN model architecture
│   ├── training/           # Training pipeline
│   └── evaluation/         # Evaluation & visualization
├── scripts/                # Command-line scripts
│   ├── train.py           # Training script
│   ├── predict.py         # Prediction script
│   └── deploy.py          # Deployment script
├── notebooks/             # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
├── app/                   # Web application
│   ├── app.py            # Flask server
│   ├── templates/        # HTML templates
│   └── static/           # Static files
├── artifacts/            # Generated files
│   ├── models/          # Saved models
│   ├── logs/           # Training logs
│   └── results/        # Plots & metrics
└── tests/               # Unit tests
🔧 Configuration
Edit config/config.yaml to customize:

Model architecture (layers, filters, dropout)

Training parameters (epochs, batch size, learning rate)

Data preprocessing options

Path configurations

📈 Model Performance
Training History
https://artifacts/results/training_history.png

Confusion Matrix
https://artifacts/results/confusion_matrix.png

Sample Predictions
https://artifacts/results/sample_predictions.png

🌐 Web Interface
The Flask web app provides:

Upload Interface: Drag & drop or click to upload

Drawing Canvas: Draw digits with your mouse/touch

Real-time Results: Instant prediction with confidence scores

Probability Distribution: Visual breakdown for all digits

🧪 Testing
bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
📚 Notebooks
Data Exploration: Understand MNIST dataset distribution

Model Training: Step-by-step training process

Results Analysis: In-depth performance analysis

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/improvement)

Commit changes (git commit -am 'Add new feature')

Push to branch (git push origin feature/improvement)

Create Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
MNIST Database

TensorFlow Documentation

Keras Examples

📞 Contact
Your Name - your.email@example.com
GitHub: @yourusername
Project Link: https://github.com/yourusername/deep-learning-portfolio