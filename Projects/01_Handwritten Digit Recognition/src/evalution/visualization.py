import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import seaborn as sns
import random

class VisualizationUtils:
    """Visualization utilities for model evaluation."""
    
    @staticmethod
    def plot_sample_predictions(x_data: np.ndarray,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               num_samples: int = 10,
                               title: str = "Sample Predictions"):
        """
        Plot sample predictions with true vs predicted labels.
        
        Args:
            x_data: Input images
            y_true: True labels
            y_pred: Predicted labels
            num_samples: Number of samples to display
            title: Plot title
        """
        # Randomly select samples
        indices = random.sample(range(len(x_data)), min(num_samples, len(x_data)))
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            img = x_data[idx].squeeze()
            
            # Plot image
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            # Set title with color coding
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            
            if true_label == pred_label:
                color = 'green'
                title_text = f"True: {true_label}\nPred: {pred_label}"
            else:
                color = 'red'
                title_text = f"True: {true_label}\nPred: {pred_label} (WRONG)"
            
            ax.set_title(title_text, color=color, fontsize=10)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_misclassified_samples(x_test: np.ndarray,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  num_samples: int = 10):
        """
        Plot misclassified samples.
        
        Args:
            x_test: Test images
            y_true: True labels
            y_pred: Predicted labels
            num_samples: Number of samples to display
        """
        # Find misclassified indices
        misclassified = np.where(y_true != y_pred)[0]
        
        if len(misclassified) == 0:
            print("No misclassified samples!")
            return
        
        # Limit number of samples
        num_samples = min(num_samples, len(misclassified))
        indices = misclassified[:num_samples]
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            ax = axes[i]
            img = x_test[idx].squeeze()
            
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", 
                        color='red', fontsize=10)
        
        plt.suptitle(f"Misclassified Samples (Total: {len(misclassified)})", 
                    fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_maps(model, 
                         image: np.ndarray,
                         layer_names: List[str] = None,
                         max_features: int = 16):
        """
        Plot feature maps from convolutional layers.
        
        Args:
            model: Keras model
            image: Input image
            layer_names: Names of layers to visualize
            max_features: Maximum number of features to display per layer
        """
        from tensorflow.keras.models import Model
        
        if layer_names is None:
            # Get convolutional layers
            layer_names = [layer.name for layer in model.layers 
                          if 'conv' in layer.name]
        
        # Create models to get intermediate outputs
        outputs = [model.get_layer(name).output for name in layer_names]
        visualization_model = Model(inputs=model.input, outputs=outputs)
        
        # Get feature maps
        feature_maps = visualization_model.predict(np.expand_dims(image, axis=0))
        
        # Plot feature maps
        for layer_name, fmap in zip(layer_names, feature_maps):
            print(f"Layer: {layer_name}, Feature map shape: {fmap.shape}")
            
            # Number of features in the feature map
            num_features = fmap.shape[-1]
            size = fmap.shape[1]
            
            # Display grid of feature maps
            display_grid = np.zeros((size, size * num_features))
            
            for i in range(min(num_features, max_features)):
                # Post-process feature to make it visually palatable
                x = fmap[0, :, :, i]
                x -= x.mean()
                x /= (x.std() + 1e-5)
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                
                # Put in grid
                display_grid[:, i * size: (i + 1) * size] = x
            
            # Display grid
            scale = 20. / num_features
            plt.figure(figsize=(scale * num_features, scale))
            plt.title(f'Feature maps for {layer_name}')
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()