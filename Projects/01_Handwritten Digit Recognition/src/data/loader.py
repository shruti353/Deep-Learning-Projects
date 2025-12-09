import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MNISTDataLoader:
    """MNIST dataset loader and preprocessor."""
    
    def __init__(self, config: dict):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.paths_config = config['paths']
        
    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                 Tuple[np.ndarray, np.ndarray]]:
        """
        Load MNIST dataset.
        
        Returns:
            (x_train, y_train), (x_test, y_test)
        """
        logger.info("Loading MNIST dataset...")
        
        # Load dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        logger.info(f"Training samples: {x_train.shape[0]}")
        logger.info(f"Test samples: {x_test.shape[0]}")
        logger.info(f"Image shape: {x_train.shape[1:]}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def preprocess_data(self, 
                       x_train: np.ndarray, 
                       x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess images: normalize, reshape, add channel dimension.
        
        Args:
            x_train: Training images
            x_test: Test images
            
        Returns:
            Preprocessed training and test images
        """
        logger.info("Preprocessing data...")
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape to include channel dimension
        img_height, img_width = self.data_config['image_shape'][:2]
        x_train = x_train.reshape(-1, img_height, img_width, 1)
        x_test = x_test.reshape(-1, img_height, img_width, 1)
        
        logger.info(f"Preprocessed shape - X_train: {x_train.shape}")
        logger.info(f"Preprocessed shape - X_test: {x_test.shape}")
        
        return x_train, x_test
    
    def create_data_generators(self, 
                              x_train: np.ndarray, 
                              y_train: np.ndarray,
                              x_val: Optional[np.ndarray] = None,
                              y_val: Optional[np.ndarray] = None):
        """
        Create data generators with optional augmentation.
        
        Args:
            x_train: Training images
            y_train: Training labels
            x_val: Validation images (optional)
            y_val: Validation labels (optional)
            
        Returns:
            train_generator, val_generator (if validation data provided)
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            validation_split=self.data_config['validation_split']
        )
        
        # For validation/test - only normalization
        test_datagen = ImageDataGenerator()
        
        batch_size = self.data_config['batch_size']
        
        # Training generator
        train_generator = train_datagen.flow(
            x_train, y_train,
            batch_size=batch_size,
            subset='training',
            shuffle=self.data_config['shuffle']
        )
        
        if x_val is not None and y_val is not None:
            # Validation generator
            val_generator = train_datagen.flow(
                x_val, y_val,
                batch_size=batch_size,
                subset='validation',
                shuffle=False
            )
            return train_generator, val_generator
        
        return train_generator
    
    def split_validation(self, 
                        x_train: np.ndarray, 
                        y_train: np.ndarray):
        """
        Split training data into train and validation sets.
        
        Args:
            x_train: Training images
            y_train: Training labels
            
        Returns:
            x_train, y_train, x_val, y_val
        """
        from sklearn.model_selection import train_test_split
        
        split_ratio = self.data_config['validation_split']
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train,
            test_size=split_ratio,
            stratify=y_train,
            random_state=42
        )
        
        logger.info(f"Training samples after split: {x_train.shape[0]}")
        logger.info(f"Validation samples: {x_val.shape[0]}")
        
        return x_train, y_train, x_val, y_val