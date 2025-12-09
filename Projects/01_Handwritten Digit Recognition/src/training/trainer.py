import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, Optional, Tuple
import os
import logging
import yaml

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training with callbacks and logging."""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config['training']
        self.paths_config = config['paths']
        
        # Create directories if they don't exist
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.paths_config['model_dir'],
            self.paths_config['log_dir'],
            self.paths_config['results_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def get_callbacks(self) -> list:
        """
        Get training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.paths_config['model_dir'],
            'best_model.h5'
        )
        
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.training_config['model_checkpoint']['monitor'],
            save_best_only=self.training_config['model_checkpoint']['save_best_only'],
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping_config = self.training_config.get('early_stopping', {})
        if early_stopping_config:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor=early_stopping_config['monitor'],
                patience=early_stopping_config['patience'],
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping_callback)
        
        # TensorBoard
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.paths_config['log_dir'],
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_callback)
        
        # Learning rate scheduler
        def lr_scheduler(epoch, lr):
            if epoch > 5:
                return lr * 0.1
            return lr
        
        lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
        callbacks.append(lr_callback)
        
        # CSV Logger
        csv_logger = keras.callbacks.CSVLogger(
            os.path.join(self.paths_config['log_dir'], 'training_log.csv')
        )
        callbacks.append(csv_logger)
        
        logger.info(f"Created {len(callbacks)} callbacks")
        return callbacks
    
    def train(self, 
              model: keras.Model,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              use_generator: bool = False) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            model: Keras model to train
            x_train: Training images
            y_train: Training labels
            x_val: Validation images (optional)
            y_val: Validation labels (optional)
            use_generator: Whether to use data generators
            
        Returns:
            Training history
        """
        logger.info("Starting model training...")
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Training parameters
        epochs = self.training_config['epochs']
        batch_size = self.config['data']['batch_size']
        
        if use_generator:
            # Train using data generators
            from src.data.loader import MNISTDataLoader
            data_loader = MNISTDataLoader(self.config)
            
            if x_val is not None and y_val is not None:
                train_gen, val_gen = data_loader.create_data_generators(
                    x_train, y_train, x_val, y_val
                )
                
                steps_per_epoch = len(x_train) // batch_size
                validation_steps = len(x_val) // batch_size
                
                history = model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                train_gen = data_loader.create_data_generators(x_train, y_train)
                steps_per_epoch = len(x_train) // batch_size
                
                history = model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )
        else:
            # Train using arrays
            if x_val is not None and y_val is not None:
                history = model.fit(
                    x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks,
                    verbose=1,
                    shuffle=True
                )
            else:
                history = model.fit(
                    x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1,
                    shuffle=True
                )
        
        logger.info("Training completed!")
        
        # Save final model
        final_model_path = os.path.join(
            self.paths_config['model_dir'],
            'final_model.h5'
        )
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Save training config
        config_path = os.path.join(
            self.paths_config['model_dir'],
            'training_config.yaml'
        )
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        return history