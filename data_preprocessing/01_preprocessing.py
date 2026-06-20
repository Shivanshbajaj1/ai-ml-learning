"""
03. Data Preprocessing
Install: pip install pandas scikit-learn numpy
Run directly: python 01_preprocessing.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


def sample_data():
    return pd.DataFrame({
        "age": [25, 30, np.nan, 28, 35, 40],
        "salary": [50000, 60000, 55000, np.nan, 80000, 90000],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA", "Chicago"],
        "purchased": [0, 1, 0, 1, 1, 0],   # target variable
    })


def handle_missing_values(df):
    df = df.copy()
    print("Missing values before:\n", df.isna().sum())

    # Impute numeric columns with the median (robust to outliers, unlike mean)
    df["age"] = df["age"].fillna(df["age"].median())
    df["salary"] = df["salary"].fillna(df["salary"].median())

    print("\nMissing values after:\n", df.isna().sum())
    return df


def encode_categorical(df):
    df = df.copy()

    # Label encoding: turns categories into integers (0, 1, 2, ...).
    # Fine for the TARGET column, risky for FEATURES (implies false ordering).
    le = LabelEncoder()
    df["purchased_encoded"] = le.fit_transform(df["purchased"])

    # One-hot encoding: turns categories into separate 0/1 columns.
    # Correct choice for nominal (unordered) FEATURE columns like "city".
    city_dummies = pd.get_dummies(df["city"], prefix="city")
    df = pd.concat([df, city_dummies], axis=1)

    print("\nAfter encoding:\n", df)
    return df


def scale_features(df, columns):
    """
    StandardScaler transforms each column to mean=0, std=1.
    Important for distance-based models (KNN, SVM) and gradient descent
    (features on wildly different scales slow down / distort training).
    """
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print(f"\nScaled columns {columns}:\n", df[columns])
    return df


def train_test_split_example(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 80/20 split, random_state fixes the shuffle so results are reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = sample_data()
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = scale_features(df, ["age", "salary"])
    train_test_split_example(df, target_col="purchased")
