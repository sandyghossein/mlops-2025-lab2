import pandas as pd
import numpy as np
from typing import Optional
from .base_features_computer import BaseFeaturesComputer


class TitanicFeaturesComputer(BaseFeaturesComputer):
    """
    Feature engineering for Titanic dataset.
    
    Creates:
    - FamilySize: SibSp + Parch + 1
    - Title: Extracted from Name
    - One-hot encoding for categorical variables
    """
    
    def __init__(self):
        """Initialize the features computer."""
        super().__init__()
        print("✓ TitanicFeaturesComputer initialized")
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for Titanic dataset."""
        df = df.copy()

        # 1) Fill numeric missing values
        if "Fare" in df.columns:
            df["Fare"] = df["Fare"].fillna(df["Fare"].median())

        if "Age" in df.columns:
            df["Age"] = df["Age"].fillna(df["Age"].median())

        # 2) Family size
        if {"SibSp", "Parch"}.issubset(df.columns):
            df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1

        # 3) Title extraction from Name
        if "Name" in df.columns:
            df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).astype(str).str.strip()
            # Group rare titles
            common_titles = ["Mr", "Mrs", "Miss", "Master"]
            df["Title"] = df["Title"].where(df["Title"].isin(common_titles), other="Rare")

        # 4) One-hot encoding for categorical variables
        cat_cols = []
        for c in ["Sex", "Embarked", "Pclass", "Title"]:
            if c in df.columns:
                cat_cols.append(c)

        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # 5) Drop non-feature columns
        drop_if_exists = ["Name", "Ticket", "Cabin", "PassengerId"]
        for c in drop_if_exists:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

        # 6) Fill any remaining NaNs
        df = df.fillna(0)

        return df
    
    def compute(self, df: pd.DataFrame, mode: str = "train", 
                feature_columns: Optional[list] = None) -> tuple[pd.DataFrame, Optional[list]]:
        """
        Compute features based on mode.
        
        Args:
            df: Input DataFrame
            mode: "train" or "inference"
            feature_columns: Feature columns from training (for inference mode)
            
        Returns:
            Tuple of (features_df, feature_columns_list)
        """
        print(f"Computing features in {mode} mode...")
        
        # Create features
        fe_df = self._create_features(df)
        
        if mode == "train":
            # Training mode: save feature schema
            if "Survived" not in fe_df.columns:
                raise ValueError("train mode expects 'Survived' column in input CSV")
            
            target = fe_df["Survived"].astype(int)
            X = fe_df.drop(columns=["Survived"])
            
            # Get feature columns
            feature_cols = list(X.columns)
            
            # Combine features and target
            out_df = pd.concat([X, target.rename("Survived")], axis=1)
            
            print(f"✓ Created {len(feature_cols)} features")
            return out_df, feature_cols
            
        elif mode == "inference":
            # Inference mode: align with training features
            if feature_columns is None:
                raise ValueError("inference mode requires feature_columns")
            
            # Align columns with training
            X = fe_df.reindex(columns=feature_columns, fill_value=0)
            
            print(f"✓ Aligned to {len(feature_columns)} training features")
            return X, feature_columns
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'")