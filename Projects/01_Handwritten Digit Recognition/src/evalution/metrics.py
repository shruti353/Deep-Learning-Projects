import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
        """
        self.model = model
    
    def evaluate(self, 
                x_test: np.ndarray, 
                y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            x_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        
        # Get predictions
        y_pred_probs = self.model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        
        # Keras evaluation
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'sklearn_accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        return metrics
    
    def get_predictions(self, 
                       x_data: np.ndarray, 
                       return_probs: bool = False) -> np.ndarray:
        """
        Get model predictions.
        
        Args:
            x_data: Input data
            return_probs: Whether to return probabilities or class labels
            
        Returns:
            Predictions
        """
        predictions = self.model.predict(x_data, verbose=0)
        
        if return_probs:
            return predictions
        else:
            return np.argmax(predictions, axis=1)
    
    def analyze_errors(self, 
                      x_test: np.ndarray, 
                      y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze misclassified samples.
        
        Args:
            x_test: Test images
            y_test: Test labels
            
        Returns:
            misclassified_indices, true_labels, predicted_labels
        """
        y_pred = self.get_predictions(x_test)
        
        # Find misclassified samples
        misclassified = y_pred != y_test
        misclassified_indices = np.where(misclassified)[0]
        
        logger.info(f"Number of misclassified samples: {len(misclassified_indices)}")
        logger.info(f"Error rate: {len(misclassified_indices)/len(y_test):.4f}")
        
        return misclassified_indices, y_test[misclassified_indices], y_pred[misclassified_indices]