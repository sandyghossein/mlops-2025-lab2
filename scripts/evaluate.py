"""
Evaluation script with CLI.

Usage:
    python scripts/evaluate.py --model models/model.pkl \
                               --input data/features/train.csv \
                               --metrics_output results/metrics.json
"""

import argparse
import json
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mlops_2025.models import LogisticModel


def main(model_path: str, data_path: str, metrics_output: str | None, model_type: str = "logistic"):
    """
    Main evaluation function.
    
    Args:
        model_path: Path to saved model
        data_path: Path to evaluation data
        metrics_output: Path to save metrics JSON
        model_type: Type of model
    """
    print(f"Loading model from: {model_path}")
    
    # Load model based on type
    if model_type == "logistic":
        model = LogisticModel()
    else:
        raise NotImplementedError(f"Model type '{model_type}' not yet implemented")
    
    model.load(model_path)

    print(f"Loading evaluation data from: {data_path}")
    df = pd.read_csv(data_path)

    if "Survived" not in df.columns:
        raise ValueError("Evaluation CSV must contain 'Survived' column")

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    print("Computing metrics...")
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    
    metrics = {"accuracy": float(accuracy)}

    print("\n" + "="*50)
    print(f"ACCURACY: {accuracy:.4f}")
    print("="*50 + "\n")
    
    print("Classification Report:")
    print(classification_report(y, preds, target_names=['Not Survived', 'Survived']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds))

    if metrics_output:
        out_path = Path(metrics_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved to: {metrics_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", required=True, help="Path to saved model")
    parser.add_argument("--input", required=True, help="Path to evaluation CSV")
    parser.add_argument("--metrics_output", required=False, help="Path to save metrics JSON")
    parser.add_argument("--model-type", default="logistic", 
                        choices=["logistic"], help="Type of model")

    args = parser.parse_args()
    main(args.model, args.input, args.metrics_output, args.model_type)