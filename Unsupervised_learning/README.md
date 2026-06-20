# 05. Unsupervised Learning

## Concept

No labels — the algorithm finds structure in the data on its own. Most common
task: **clustering** (grouping similar points together).

## K-Means Clustering

1. Pick `k` random points as initial cluster centers (centroids).
2. Assign every point to its nearest centroid.
3. Recompute each centroid as the mean of its assigned points.
4. Repeat steps 2–3 until centroids stop moving (convergence).

Choosing `k`: the **elbow method** — plot inertia (within-cluster sum of
squares) for different `k` values and look for where it stops dropping
sharply (the "elbow").

## Files

- `01_kmeans_clustering.py` — K-Means from scratch, then scikit-learn,
  plus the elbow method
