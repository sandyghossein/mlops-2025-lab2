"""
Training script with CLI.

Usage:
    python scripts/train.py --input data/features/train.csv \
                            --model-type logistic \
                            --output models/model.pkl
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mlops_2025.models import LogisticModel


def main(input_csv: str, output_model: str, model_type: str = "logistic"):
    """
    Main training function.
    
    Args:
        input_csv: Path to features CSV
        output_model: Path to save trained model
        model_type: Type of model to train
    """
    # Load engineered features
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Separate features and target
    if "Survived" not in df.columns:
        raise ValueError("Target column 'Survived' not found")
    
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Initialize model based on type
    if model_type == "logistic":
        model = LogisticModel(max_iter=500)
    else:
        raise NotImplementedError(f"Model type '{model_type}' not yet implemented")

    # Train model
    model.train(X, y)

    # Save model
    model.save(output_model)
    print(f"✅ Model trained and saved to {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML Model")
    parser.add_argument("--input", required=True, help="Path to features CSV")
    parser.add_argument("--output", required=True, help="Path to save trained model")
    parser.add_argument("--model-type", default="logistic", 
                        choices=["logistic"], help="Type of model")
    
    args = parser.parse_args()
    main(args.input, args.output, args.model_type)