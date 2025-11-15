import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest implementation.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
            **kwargs: Additional RandomForestClassifier parameters
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state, 
            **kwargs
        )
        print(f"✓ RandomForestModel initialized (n_estimators={n_estimators})")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the random forest model."""
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
            "model_type": "RandomForest",
            "n_estimators": self.n_estimators,
            "random_state": self.random_state
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
            # Handle legacy pickles
            self.model = payload
            self.feature_columns = None
            print(f"✓ Model loaded from {path}")