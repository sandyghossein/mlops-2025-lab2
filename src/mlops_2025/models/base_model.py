from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for ML models.
    
    Allows easy swapping of different model implementations.
    """
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.feature_columns = None
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target series
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass