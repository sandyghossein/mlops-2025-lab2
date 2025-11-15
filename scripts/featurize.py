"""
Feature engineering script with CLI.

Usage:
    python scripts/featurize.py --input data/processed/train.csv \
                                 --output data/features/train.csv \
                                 --mode train
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import sys

# Add src to path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mlops_2025.features import TitanicFeaturesComputer

FEATURE_COLUMNS_FILE = "data/processed/feature_columns.json"


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    p = argparse.ArgumentParser(description="Featurize dataset (train or inference mode)")
    p.add_argument("--input", required=True, help="Path to processed CSV")
    p.add_argument("--output", required=True, help="Path to save featurized CSV")
    p.add_argument("--mode", required=True, choices=["train", "inference"], 
                   help="train or inference mode")
    p.add_argument("--feature-columns-file", default=FEATURE_COLUMNS_FILE,
                   help="Path to save/load feature columns JSON")
    return p


def main():
    """Main function using TitanicFeaturesComputer class."""
    args = build_parser().parse_args()
    inp = Path(args.input)
    outp = Path(args.output)
    feat_file = Path(args.feature_columns_file)

    # Load data
    df = pd.read_csv(inp)
    mode = args.mode

    # Create feature computer instance
    feature_computer = TitanicFeaturesComputer()

    if mode == "train":
        # Compute features and save schema
        fe_df, feature_columns = feature_computer.compute(df, mode="train")
        
        # Save feature columns for inference later
        feat_file.parent.mkdir(parents=True, exist_ok=True)
        with open(feat_file, "w") as f:
            json.dump(feature_columns, f)
        
        # Save features
        outp.parent.mkdir(parents=True, exist_ok=True)
        fe_df.to_csv(outp, index=False)
        
        print(f"✓ Saved {len(feature_columns)} features to {outp}")
        print(f"✓ Feature columns saved to {feat_file}")

    elif mode == "inference":
        # Load feature columns from training
        if not feat_file.exists():
            raise FileNotFoundError(
                f"Feature columns file not found: {feat_file}. Run train mode first."
            )
        
        with open(feat_file, "r") as f:
            feature_columns = json.load(f)

        # Compute features aligned with training
        fe_df, _ = feature_computer.compute(df, mode="inference", 
                                           feature_columns=feature_columns)
        
        # Save
        outp.parent.mkdir(parents=True, exist_ok=True)
        fe_df.to_csv(outp, index=False)
        
        print(f"✓ Saved {len(feature_columns)} features to {outp}")


if __name__ == "__main__":
    main()