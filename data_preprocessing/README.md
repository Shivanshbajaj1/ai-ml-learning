# 03. Data Preprocessing

## Why it matters

"Garbage in, garbage out." Real-world data is messy: missing values, wrong
scales, categorical columns models can't read directly. Most ML project time
goes here, not into the model itself.

## Core steps

1. **Handle missing values** — drop or impute (fill with mean/median/mode).
2. **Encode categorical variables** — models need numbers, not strings
   (one-hot encoding, label encoding).
3. **Scale numeric features** — many algorithms (KNN, gradient descent-based
   models, neural nets) are sensitive to feature scale.
4. **Split data** — train/test split, so you evaluate on data the model has
   never seen.

## Files

- `01_preprocessing.py` — all four steps using pandas + scikit-learn
