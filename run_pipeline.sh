#!/usr/bin/env bash
set -euo pipefail

RAW_TRAIN="data/train.csv"
RAW_TEST="data/test.csv"
PROCESSED_TRAIN="data/processed/train_processed.csv"
PROCESSED_TEST="data/processed/test_processed.csv"
FEATURES_TRAIN="data/features/train_features.csv"
FEATURES_UNLABELED="data/features/unlabeled.csv"
MODEL_OUT="models/model.pkl"
EVAL_OUT="results/eval.json"
PRED_OUT="results/predictions.csv"

mkdir -p data/processed data/features models results

echo "1) Preprocess"
python3 scripts/preprocess.py --train_path "$RAW_TRAIN" --test_path "$RAW_TEST" \
    --output_train "$PROCESSED_TRAIN" --output_test "$PROCESSED_TEST"

echo "2) Featurize (train)"
python3 scripts/featurize.py --input "$PROCESSED_TRAIN" --output "$FEATURES_TRAIN" --mode train

echo "3) Train"
python3 scripts/train.py --input "$FEATURES_TRAIN" --output "$MODEL_OUT"

echo "4) Evaluate"
python3 scripts/evaluate.py --model "$MODEL_OUT" --input "$FEATURES_TRAIN" --output "$EVAL_OUT"

echo "5) Featurize test + Predict"
python3 scripts/featurize.py --input "$PROCESSED_TEST" --output "$FEATURES_UNLABELED" --mode inference
python3 scripts/predict.py --model "$MODEL_OUT" --input "$FEATURES_UNLABELED" --output "$PRED_OUT"

echo "Done."
