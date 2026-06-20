"""
02. Pandas Basics
Install: pip install pandas
Run directly: python 02_pandas_basics.py
"""
import pandas as pd
import numpy as np


def dataframe_creation():
    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "score": [85.5, 90.2, 78.1, 92.0],
    }
    df = pd.DataFrame(data)
    print(df)
    print("\nshape:", df.shape)
    print("dtypes:\n", df.dtypes)
    return df


def selecting_and_filtering(df):
    print("\nselect 'name' column:\n", df["name"])
    print("\nrows where age > 27:\n", df[df["age"] > 27])
    print("\nfirst 2 rows:\n", df.head(2))


def handling_missing_data():
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": [np.nan, 2, 3, 4],
    })
    print("\nwith missing values:\n", df)
    print("\nisna():\n", df.isna())
    print("\nfillna(0):\n", df.fillna(0))
    print("\ndropna():\n", df.dropna())


def groupby_aggregation():
    df = pd.DataFrame({
        "department": ["Eng", "Eng", "Sales", "Sales", "Eng"],
        "salary": [70000, 80000, 60000, 65000, 75000],
    })
    print("\ngroupby department, mean salary:\n", df.groupby("department")["salary"].mean())
    print("\ngroupby department, count:\n", df.groupby("department").size())


if __name__ == "__main__":
    df = dataframe_creation()
    selecting_and_filtering(df)
    handling_missing_data()
    groupby_aggregation()
