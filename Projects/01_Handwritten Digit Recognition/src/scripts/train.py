#!/usr/bin/env python3
"""
Main training script for Handwritten Digit Recognition.
"""

import os
import sys
import yaml
import logging
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import MNISTDataLoader
from src.models.cnn_model import DigitRecognizerCNN
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualizations import VisualizationUtils
from src.models.utils import ModelUtils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(level