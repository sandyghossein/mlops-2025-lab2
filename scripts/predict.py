"""
Prediction script with CLI.

Usage:
    python scripts/predict.py --model models/model.pkl \
                              --input data/features/test.csv \
                              --output results/predictions.csv
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mlops_2025.models import LogisticModel


def main(model_path: str, features_path: str, output_path: str, 
         model_type: str = "logistic", id_column: str | None = None):
    """
    Main prediction function.
    
    Args:
        model_path: Path to trained model
        features_path: Path to features CSV
        output_path: Path to save predictions
        model_type: Type of model
        id_column: Optional ID column to include
    """
    # Check paths
    model_path = Path(model_path)
    features_path = Path(features_path)
    output_path = Path(output_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    # Load model
    print(f"Loading model from: {model_path}")
    if model_type == "logistic":
        model = LogisticModel()
    else:
        raise NotImplementedError(f"Model type '{model_type}' not yet implemented")
    
    model.load(model_path)

    # Load features
    print(f"Loading features from: {features_path}")
    X = pd.read_csv(features_path)
    print(f"Features shape: {X.shape}")

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X)
    print(f"✓ Generated {len(predictions)} predictions")

    # Create output DataFrame
    output_df = pd.DataFrame({"prediction": predictions})
    
    # Add ID column if specified
    if id_column and id_column in X.columns:
        output_df.insert(0, id_column, X[id_column])

    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")
    print("✅ Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (pickle)")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV with features")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions CSV")
    parser.add_argument("--model-type", default="logistic", 
                        choices=["logistic"], help="Type of model")
    parser.add_argument("--id-column", type=str, required=False, 
                        help="Optional ID column to include in output")
    
    args = parser.parse_args()
    main(args.model, args.input, args.output, args.model_type, args.id_column)