#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
import pandas as pd
import sys

# -------------------------------
# Load Model (support dict payload or raw model)
# -------------------------------
def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    # payload could be {"model": model_obj, "feature_columns": [...]}
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_columns = payload.get("feature_columns", None)
        print(f"✅ Model payload loaded from: {model_path} (contains model + feature metadata)")
    else:
        model = payload
        feature_columns = None
        print(f"✅ Model loaded from: {model_path}")
    return model, feature_columns

# -------------------------------
# Load Features
# -------------------------------
def load_features(features_path: Path):
    X = pd.read_csv(features_path)
    print(f"📊 Loaded features with shape: {X.shape}")
    return X

# -------------------------------
# Predict
# -------------------------------
def make_predictions(model, X: pd.DataFrame):
    y_pred = model.predict(X)
    print(f"🪄 Generated {len(y_pred)} predictions.")
    return y_pred

# -------------------------------
# Save Predictions
# -------------------------------
def save_predictions(predictions, output_path: Path, X: pd.DataFrame, id_column: str | None = None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"prediction": predictions})
    if id_column and id_column in X.columns:
        out_df.insert(0, id_column, X[id_column])
    out_df.to_csv(output_path, index=False)
    print(f"💾 Predictions saved to: {output_path}")

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference using a trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (pickle)")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV with features (no labels)")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions CSV")
    parser.add_argument("--id-column", type=str, required=False, help="Optional id column to include in output (e.g., PassengerId)")
    args = parser.parse_args()

    model_path = Path(args.model)
    features_path = Path(args.input)
    output_path = Path(args.output)

    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        sys.exit(2)
    if not features_path.exists():
        print(f"Features file not found: {features_path}", file=sys.stderr)
        sys.exit(2)

    model, feature_columns = load_model(model_path)
    X = load_features(features_path)

    # Reindex to training feature order if available
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
        print(f"🔁 Reindexed features to training order ({len(feature_columns)} cols).")

    preds = make_predictions(model, X)
    save_predictions(preds, output_path, X, args.id_column)

    print("✅ Inference complete!")

if __name__ == "__main__":
    main()

