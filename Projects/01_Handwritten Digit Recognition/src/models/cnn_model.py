import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DigitRecognizerCNN:
    """CNN model for handwritten digit recognition."""
    
    def __init__(self, config: Dict):
        """
        Initialize CNN model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data']
        self.model = None
        
    def build_model(self) -> keras.Model:
        """
        Build CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building CNN model...")
        
        # Input layer
        input_shape = tuple(self.data_config['image_shape'])
        inputs = keras.Input(shape=input_shape)
        
        # Feature extraction layers
        x = self._build_feature_extractor(inputs)
        
        # Classification layers
        outputs = self._build_classification_head(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name="digit_cnn")
        
        # Compile model
        self._compile_model(model)
        
        self.model = model
        logger.info("Model built successfully!")
        model.summary(print_fn=logger.info)
        
        return model
    
    def _build_feature_extractor(self, x: layers.Layer) -> layers.Layer:
        """Build convolutional feature extraction layers."""
        model_config = self.model_config
        
        # First convolutional block
        x = layers.Conv2D(
            filters=model_config['layers']['conv1']['filters'],
            kernel_size=model_config['layers']['conv1']['kernel_size'],
            activation=model_config['layers']['conv1']['activation'],
            padding=model_config['layers']['conv1']['padding'],
            name="conv1"
        )(x)
        
        if model_config.get('use_batch_norm', False):
            x = layers.BatchNormalization()(x)
        
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
        
        # Second convolutional block
        x = layers.Conv2D(
            filters=model_config['layers']['conv2']['filters'],
            kernel_size=model_config['layers']['conv2']['kernel_size'],
            activation=model_config['layers']['conv2']['activation'],
            padding=model_config['layers']['conv2']['padding'],
            name="conv2"
        )(x)
        
        if model_config.get('use_batch_norm', False):
            x = layers.BatchNormalization()(x)
        
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
        
        # Flatten layer
        x = layers.Flatten(name="flatten")(x)
        
        return x
    
    def _build_classification_head(self, x: layers.Layer) -> layers.Layer:
        """Build classification head layers."""
        model_config = self.model_config
        
        # Dense layer
        x = layers.Dense(
            units=model_config['layers']['dense']['units'],
            activation=model_config['layers']['dense']['activation'],
            name="dense1"
        )(x)
        
        # Dropout for regularization
        dropout_rate = model_config.get('dropout_rate', 0.5)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="dropout")(x)
        
        # Output layer
        outputs = layers.Dense(
            units=self.data_config['num_classes'],
            activation='softmax',
            name="output"
        )(x)
        
        return outputs
    
    def _compile_model(self, model: keras.Model):
        """Compile the model with optimizer, loss, and metrics."""
        training_config = self.config['training']
        
        # Optimizer
        optimizer_name = training_config['optimizer'].lower()
        learning_rate = training_config['learning_rate']
        
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(
                learning_rate=learning_rate, 
                momentum=0.9
            )
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Loss function
        loss = training_config['loss']
        
        # Metrics
        metrics = training_config['metrics']
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def get_model(self) -> keras.Model:
        """Get the model instance."""
        if self.model is None:
            self.build_model()
        return self.model