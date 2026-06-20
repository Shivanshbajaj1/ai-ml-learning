"""
02. Matplotlib Basics
Install: pip install matplotlib numpy
Run directly: python 01_matplotlib_basics.py
(Saves plots to PNG files in this folder instead of popping up a window,
so it works in headless/server environments too.)
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for headless environments
import matplotlib.pyplot as plt
import numpy as np


def line_plot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label="sin(x)")
    plt.title("Line Plot")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    plt.savefig("01_line_plot.png")
    plt.close()


def histogram():
    data = np.random.normal(loc=50, scale=10, size=1000)

    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=30, color="steelblue", edgecolor="black")
    plt.title("Histogram — Distribution of a Normal Variable")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("02_histogram.png")
    plt.close()


def scatter_plot():
    x = np.random.rand(100) * 10
    y = 2 * x + np.random.randn(100) * 2   # roughly linear with noise

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6)
    plt.title("Scatter Plot — Relationship Between Two Variables")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("03_scatter.png")
    plt.close()


def bar_chart():
    categories = ["Python", "Java", "C++", "JavaScript"]
    counts = [45, 25, 15, 35]

    plt.figure(figsize=(6, 4))
    plt.bar(categories, counts, color="coral")
    plt.title("Bar Chart — Comparing Categories")
    plt.ylabel("Count")
    plt.savefig("04_bar_chart.png")
    plt.close()


def box_plot():
    data = [np.random.normal(0, std, 100) for std in [1, 2, 3]]

    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=["low variance", "medium variance", "high variance"])
    plt.title("Box Plot — Spread and Outliers")
    plt.savefig("05_box_plot.png")
    plt.close()


if __name__ == "__main__":
    line_plot()
    histogram()
    scatter_plot()
    bar_chart()
    box_plot()
    print("Saved 5 PNG files to this folder.")
