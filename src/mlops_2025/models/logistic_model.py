import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from .base_model import BaseModel


class LogisticModel(BaseModel):
    """
    Logistic Regression implementation.
    """
    
    def __init__(self, max_iter: int = 500, **kwargs):
        """
        Initialize Logistic Regression model.
        
        Args:
            max_iter: Maximum iterations for solver
            **kwargs: Additional LogisticRegression parameters
        """
        super().__init__()
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.model = LogisticRegression(max_iter=max_iter, **kwargs)
        print(f"✓ LogisticModel initialized (max_iter={max_iter})")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the logistic regression model."""
        self.feature_columns = list(X.columns)
        
        print(f"Training on {len(X)} samples with {len(self.feature_columns)} features...")
        self.model.fit(X, y)
        print("✓ Training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """Save model and metadata."""
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "model_type": "LogisticRegression",
            "max_iter": self.max_iter
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model and metadata."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        
        if isinstance(payload, dict):
            self.model = payload["model"]
            self.feature_columns = payload.get("feature_columns", None)
            print(f"✓ Model loaded from {path} (with metadata)")
        else:
            # Handle legacy pickles (just the model)
            self.model = payload
            self.feature_columns = None
            print(f"✓ Model loaded from {path}")