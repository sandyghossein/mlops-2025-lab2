import pandas as pd
import numpy as np
import warnings
from .base_preprocessor import BasePreprocessor

warnings.filterwarnings("ignore")


class TitanicPreprocessor(BasePreprocessor):
    """
    Concrete implementation of preprocessing for Titanic dataset.
    
    Handles:
    - Missing values
    - Dropping unnecessary columns
    - Data cleaning
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        super().__init__()
        print("✓ TitanicPreprocessor initialized")
    
    def _clean_data(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values.
        
        Args:
            train: Training DataFrame
            test: Test DataFrame
            
        Returns:
            Unified cleaned DataFrame
        """
        # Make copies to avoid modifying originals
        train = train.copy()
        test = test.copy()
        
        # Drop Cabin column due to numerous null values
        train.drop(columns=['Cabin'], inplace=True)
        test.drop(columns=['Cabin'], inplace=True)

        # Fill missing values
        train['Embarked'].fillna('S', inplace=True)
        test['Fare'].fillna(test['Fare'].mean(), inplace=True)

        # Create unified dataframe for easier manipulation
        df = pd.concat([train, test], sort=True).reset_index(drop=True)
        
        # Fill missing Age values using group median
        df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )

        return df
    
    def _split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the unified dataframe back into train and test sets.
        
        Args:
            df: Unified DataFrame
            
        Returns:
            Tuple of (train, test)
        """
        train = df.loc[:890].copy()
        test = df.loc[891:].copy()

        # Remove Survived column from test set
        if 'Survived' in test.columns:
            test.drop(columns=['Survived'], inplace=True)

        # Ensure Survived column is int in train set
        if 'Survived' in train.columns:
            train['Survived'] = train['Survived'].astype('int64')

        return train, test
    
    def process(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main preprocessing pipeline.
        
        Args:
            train: Raw training DataFrame
            test: Raw test DataFrame
            
        Returns:
            Tuple of (processed_train, processed_test)
        """
        print(f"Processing train: {train.shape}, test: {test.shape}")
        
        # Clean data
        print("Cleaning data...")
        df_cleaned = self._clean_data(train, test)
        
        # Split back
        print("Splitting data...")
        train_processed, test_processed = self._split_data(df_cleaned)
        
        print(f"✓ Final train shape: {train_processed.shape}")
        print(f"✓ Final test shape: {test_processed.shape}")
        
        return train_processed, test_processed