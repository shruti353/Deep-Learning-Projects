import numpy as np
from typing import Tuple
import cv2
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Additional image preprocessing utilities."""
    
    @staticmethod
    def normalize_images(images: np.ndarray) -> np.ndarray:
        """Normalize images to [0, 1] range."""
        return images.astype('float32') / 255.0
    
    @staticmethod
    def standardize_images(images: np.ndarray) -> np.ndarray:
        """Standardize images to have zero mean and unit variance."""
        mean = np.mean(images)
        std = np.std(images)
        return (images - mean) / (std + 1e-7)
    
    @staticmethod
    def augment_image(image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to a single image."""
        import random
        
        # Random rotation
        angle = random.uniform(-15, 15)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Random shift
        shift_x = random.uniform(-0.1, 0.1) * width
        shift_y = random.uniform(-0.1, 0.1) * height
        shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        image = cv2.warpAffine(image, shift_matrix, (width, height))
        
        return image
    
    @staticmethod
    def resize_images(images: np.ndarray, 
                     target_size: Tuple[int, int]) -> np.ndarray:
        """Resize images to target size."""
        resized_images = []
        for img in images:
            # Remove channel dimension for OpenCV
            if len(img.shape) == 3 and img.shape[2] == 1:
                img = img[:, :, 0]
            
            resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Add channel dimension back if needed
            if len(resized.shape) == 2:
                resized = resized[:, :, np.newaxis]
            
            resized_images.append(resized)
        
        return np.array(resized_images)