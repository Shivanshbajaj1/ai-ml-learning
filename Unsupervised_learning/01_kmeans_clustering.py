"""
05. K-Means Clustering — from scratch, then scikit-learn.
Install: pip install numpy scikit-learn
Run directly: python 01_kmeans_clustering.py
"""
import numpy as np


def kmeans_from_scratch(X, k, max_iters=100, seed=42):
    """
    Implements the 4-step K-Means loop described in the README.
    Returns (centroids, labels).
    """
    rng = np.random.default_rng(seed)
    # Step 1: pick k random data points as initial centroids
    initial_idx = rng.choice(len(X), size=k, replace=False)
    centroids = X[initial_idx].copy()

    for iteration in range(max_iters):
        # Step 2: assign each point to its nearest centroid
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1) for centroid in centroids
        ])  # shape (k, n_points)
        labels = np.argmin(distances, axis=0)

        # Step 3: recompute centroids as the mean of assigned points
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        # Step 4: check for convergence
        if np.allclose(centroids, new_centroids):
            print(f"  converged after {iteration} iterations")
            break

        centroids = new_centroids

    return centroids, labels


def inertia(X, centroids, labels):
    """Within-cluster sum of squares — lower is tighter clusters. Used for the elbow method."""
    total = 0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        total += np.sum((cluster_points - centroid) ** 2)
    return total


def elbow_method(X, k_range=range(1, 7)):
    """Run K-Means for each k, print inertia so you can look for the 'elbow.'"""
    print("Elbow method (look for where inertia stops dropping sharply):")
    for k in k_range:
        centroids, labels = kmeans_from_scratch(X, k)
        print(f"  k={k}: inertia={inertia(X, centroids, labels):.2f}")


def kmeans_sklearn(X, k):
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X)
    return model.cluster_centers_, model.labels_


if __name__ == "__main__":
    # Generate 3 visually separated blobs of points
    rng = np.random.default_rng(0)
    cluster1 = rng.normal(loc=[0, 0], scale=0.5, size=(30, 2))
    cluster2 = rng.normal(loc=[5, 5], scale=0.5, size=(30, 2))
    cluster3 = rng.normal(loc=[0, 5], scale=0.5, size=(30, 2))
    X = np.vstack([cluster1, cluster2, cluster3])

    print("From scratch (k=3):")
    centroids, labels = kmeans_from_scratch(X, k=3)
    print("  centroids:\n", centroids)
    print("  cluster sizes:", np.bincount(labels))

    print("\nscikit-learn (k=3):")
    sk_centroids, sk_labels = kmeans_sklearn(X, k=3)
    print("  centroids:\n", sk_centroids)

    print()
    elbow_method(X)
