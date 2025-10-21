# scripts/featurize.py
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np

FEATURE_COLUMNS_FILE = "data/processed/feature_columns.json"

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Featurize dataset (train or inference mode)")
    p.add_argument("--input", required=True, help="Path to processed CSV (from preprocess)")
    p.add_argument("--output", required=True, help="Path to save featurized CSV")
    p.add_argument("--mode", required=True, choices=["train", "inference"], help="train or inference")
    p.add_argument("--feature-columns-file", default=FEATURE_COLUMNS_FILE,
                   help="Filepath to save/load feature columns json")
    return p

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features used in the notebook. Adjust to match your notebook exactly."""
    df = df.copy()

    # Example feature engineering for Titanic-like dataset.
    # 1) Numeric simple fills
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # 2) Family size
    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1

    # 3) Title extraction from Name (if present)
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).astype(str).str.strip()
        # group rare titles
        common_titles = ["Mr", "Mrs", "Miss", "Master"]
        df["Title"] = df["Title"].where(df["Title"].isin(common_titles), other="Rare")

    # 4) Categorical -> dummies
    cat_cols = []
    for c in ["Sex", "Embarked", "Pclass", "Title"]:
        if c in df.columns:
            cat_cols.append(c)

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 5) Optionally drop original non-feature columns (Name, Ticket, Cabin, etc.)
    drop_if_exists = ["Name", "Ticket", "Cabin", "PassengerId"]
    for c in drop_if_exists:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # 6) Ensure no NaNs remain in feature columns
    df = df.fillna(0)

    return df

def main():
    args = build_parser().parse_args()
    inp = Path(args.input)
    outp = Path(args.output)
    feat_file = Path(args.feature_columns_file)

    df = pd.read_csv(inp)
    mode = args.mode

    # create features
    fe_df = create_features(df)

    if mode == "train":
        # Expect 'Survived' target in training data
        if "Survived" not in fe_df.columns:
            raise ValueError("train mode expects 'Survived' column in input CSV")
        # Keep target as last column (or first, whichever you prefer)
        target = fe_df["Survived"].astype(int)
        X = fe_df.drop(columns=["Survived"])
        # Save feature column order for inference
        feature_columns = list(X.columns)
        feat_file.parent.mkdir(parents=True, exist_ok=True)
        with open(feat_file, "w") as f:
            json.dump(feature_columns, f)
        # Save X and y side-by-side
        out_df = pd.concat([X, target.rename("Survived")], axis=1)
        outp.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(outp, index=False)
        print(f"[train] saved featurized data with {len(feature_columns)} features to {outp}")
        print(f"[train] feature columns saved to {feat_file}")

    elif mode == "inference":
        # Load feature columns produced at training time
        if not feat_file.exists():
            raise FileNotFoundError(f"Feature columns file not found: {feat_file}. Run train mode first.")
        with open(feat_file, "r") as f:
            feature_columns = json.load(f)

        # Keep only feature columns in the same order, add missing columns with 0
        X = fe_df.reindex(columns=feature_columns, fill_value=0)
        outp.parent.mkdir(parents=True, exist_ok=True)
        X.to_csv(outp, index=False)
        print(f"[inference] saved features ({len(feature_columns)}) to {outp}")

if __name__ == "__main__":
    main()
