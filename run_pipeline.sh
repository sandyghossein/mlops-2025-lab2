#!/bin/bash
# Full ML Pipeline for Titanic Survival Prediction

set -e  # Exit on any error

echo "=========================================="
echo "Starting ML Pipeline"
echo "=========================================="

# Step 1: Preprocessing
echo ""
echo "Step 1: Preprocessing data..."
python scripts/preprocess.py \
    --train_path data/train.csv \
    --test_path data/test.csv \
    --output_train data/processed/train.csv \
    --output_test data/processed/test.csv

# Step 2: Feature Engineering (Train)
echo ""
echo "Step 2: Creating features for training data..."
python scripts/featurize.py \
    --input data/processed/train.csv \
    --output data/features/train.csv \
    --mode train

# Step 3: Feature Engineering (Test)
echo ""
echo "Step 3: Creating features for test data..."
python scripts/featurize.py \
    --input data/processed/test.csv \
    --output data/features/test.csv \
    --mode inference

# Step 4: Train Logistic Regression
echo ""
echo "Step 4: Training Logistic Regression model..."
python scripts/train.py \
    --input data/features/train.csv \
    --output models/logistic_model.pkl \
    --model-type logistic

# Step 5: Train Random Forest
echo ""
echo "Step 5: Training Random Forest model..."
python scripts/train.py \
    --input data/features/train.csv \
    --output models/rf_model.pkl \
    --model-type rf

# Step 6: Evaluate Logistic Regression
echo ""
echo "Step 6: Evaluating Logistic Regression..."
python scripts/evaluate.py \
    --model models/logistic_model.pkl \
    --input data/features/train.csv \
    --model-type logistic \
    --metrics_output results/logistic_metrics.json

# Step 7: Evaluate Random Forest
echo ""
echo "Step 7: Evaluating Random Forest..."
python scripts/evaluate.py \
    --model models/rf_model.pkl \
    --input data/features/train.csv \
    --model-type rf \
    --metrics_output results/rf_metrics.json

# Step 8: Predictions
echo ""
echo "Step 8: Making predictions on test data..."
python scripts/predict.py \
    --model models/rf_model.pkl \
    --input data/features/test.csv \
    --output results/predictions.csv \
    --model-type rf

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results:"
echo "  - Logistic metrics: results/logistic_metrics.json"
echo "  - RF metrics: results/rf_metrics.json"
echo "  - Predictions: results/predictions.csv"