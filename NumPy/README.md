# 01. Python Foundations for ML

Before any ML, you need fluency with two libraries:

## NumPy
Fast numerical arrays. Almost everything in ML (images, features, weights) is
ultimately a NumPy array under the hood. Key idea: **vectorization** — operate
on whole arrays at once instead of writing Python loops (much faster, and the
style every ML library expects).

## Pandas
Tabular data (rows/columns, like a spreadsheet) via the `DataFrame`. Used for
loading, cleaning, filtering, and exploring datasets before they go into a model.

## Files

- `01_numpy_basics.py` — arrays, vectorized ops, broadcasting, indexing
- `02_pandas_basics.py` — DataFrames, filtering, groupby, handling missing data
