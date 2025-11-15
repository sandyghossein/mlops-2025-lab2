from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class BaseFeaturesComputer(ABC):
    """
    Abstract base class for feature engineering.
    
    Supports both training mode (saves feature schema) and inference mode (uses saved schema).
    """
    
    @abstractmethod
    def compute(self, df: pd.DataFrame, mode: str = "train", 
                feature_columns: Optional[list] = None) -> tuple[pd.DataFrame, Optional[list]]:
        """
        Compute features from input DataFrame.
        
        Args:
            df: Input DataFrame
            mode: "train" or "inference"
            feature_columns: List of feature column names (for inference mode)
            
        Returns:
            Tuple of (features_df, feature_columns_list)
        """
        pass