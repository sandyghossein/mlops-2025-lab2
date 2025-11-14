"""
Preprocessing script with CLI.

Usage:
    python scripts/preprocess.py --train_path data/train.csv --test_path data/test.csv \
                                  --output_train data/processed/train.csv \
                                  --output_test data/processed/test.csv
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mlops_2025.preprocessing import TitanicPreprocessor


def main():
    """Main function using TitanicPreprocessor class."""
    parser = argparse.ArgumentParser(description='Preprocess Titanic dataset')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--output_train', type=str, required=True,
                        help='Output path for preprocessed training data')
    parser.add_argument('--output_test', type=str, required=True,
                        help='Output path for preprocessed test data')

    args = parser.parse_args()

    # Create output directories
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    print(f"Loaded train: {train.shape}, test: {test.shape}")

    # Create preprocessor and process
    preprocessor = TitanicPreprocessor()
    train_processed, test_processed = preprocessor.process(train, test)

    # Save
    print("Saving preprocessed data...")
    train_processed.to_csv(args.output_train, index=False)
    test_processed.to_csv(args.output_test, index=False)

    print(f"✓ Preprocessed train saved to: {args.output_train}")
    print(f"✓ Preprocessed test saved to: {args.output_test}")


if __name__ == "__main__":
    main()