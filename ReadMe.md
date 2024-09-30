# README

## Comparative Study of Classification Performance on Handwritten Digits

This project implements two classifiers to solve the handwritten digit classification problem using the MNIST dataset provided in `digits.mat`:

1. **Multivariate Gaussian Classifier (MLE Classifier)**
2. **k-Nearest Neighbor Classifier (KNN Classifier)**

The goal is to compare the performance of these classifiers under various conditions and analyze which one performs better.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
  - [Running the Comparison Models](#running-the-comparison-models)
  - [Testing Different Distance Metrics](#testing-different-distance-metrics)
- [Results](#results)
- [Code Overview](#code-overview)
- [References](#references)

---

## Prerequisites

- Python 3.x
- Required Python packages:
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `matplotlib`

You can install the required packages using:

```bash
pip install numpy scipy scikit-learn matplotlib
```

---

## Dataset

The dataset used is `digits.mat`, which contains:

- `X`: A 10,000 x 784 matrix, where each row represents a 28x28 pixel image of a handwritten digit, flattened into a 784-dimensional vector.
- `Y`: A 10,000 x 1 vector, where each entry is the label (0-9) corresponding to the image in `X`.

**Note:** Ensure that `digits.mat` is placed in the same directory as the script.

---

## Project Structure

- `digits.mat` - The dataset file containing the images and labels.
- `script.py` - The main Python script containing the implementation of the classifiers and the comparison code.

---

## How to Run

1. **Clone the repository or download the script and dataset to your local machine.**

2. **Ensure all prerequisites are installed.**

3. **Run the script:**

   ```bash
   python script.py
   ```

   By default, the script will execute the `runComparisonModels()` function, which compares the MLE and KNN classifiers.

### Running the Comparison Models

The `runComparisonModels()` function:

- Initializes both classifiers.
- Processes the data using PCA for dimensionality reduction.
- Splits the data into training and testing sets with various proportions.
- Trains both classifiers on the training data.
- Tests the classifiers on the testing data.
- Plots the accuracy of both classifiers against the training data proportion.

### Testing Different Distance Metrics

To test different distance metrics (L1, L2, L∞) for the KNN classifier:

1. **Comment out** the `runComparisonModels()` call.
2. **Uncomment** the `testDifferentNorms()` function call in the `__main__` section.

```python
if __name__ == "__main__":
    # runComparisonModels()
    testDifferentNorms()
```

The `testDifferentNorms()` function:

- Tests the KNN classifier with k=1 using different norms.
- Plots the accuracy for each norm against the training data proportion.

---

## Results

- **Accuracy vs. Training Data Proportion:**
  - The script will generate plots showing how the accuracy of each classifier varies with different training data sizes.
- **Distance Metrics Analysis:**
  - Another set of plots will show the performance of the KNN classifier using L1, L2, and L∞ norms.

---

## Code Overview

### Imports

```python
import numpy as np
import scipy as sp
from scipy.io import loadmat
from scipy.stats import multivariate_normal, mode
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
```

### Classes

#### `mleClassifier`

Implements the Multivariate Gaussian Classifier using Maximum Likelihood Estimation (MLE).

- **Methods:**
  - `processData()`: Performs PCA for dimensionality reduction.
  - `splitTrainingTesting(split)`: Splits the data into training and testing sets.
  - `computeClassPriors()`: Calculates prior probabilities for each class.
  - `computeClassConditionalData()`: Groups data by class.
  - `computeClassConditional()`: Calculates the mean and covariance matrix for each class.
  - `classify()`: Classifies the test data based on the highest posterior probability.
  - `computeAccuracy()`: Computes the accuracy of the classifier.
  - `reset()`: Resets the classifier's state for the next iteration.

#### `KNNClassifier`

Implements the k-Nearest Neighbor Classifier.

- **Parameters:**
  - `k`: The number of nearest neighbors to consider.

- **Methods:**
  - Similar to `mleClassifier`, with additional methods:
    - `createKDtree()`: Builds a KD-tree for efficient neighbor searches.
    - `classify(norm)`: Classifies test data using the specified norm (distance metric).

### Functions

#### `runComparisonModels()`

Runs a comparison between the MLE and KNN classifiers over different training data proportions.

#### `testDifferentNorms()`

Tests the KNN classifier with different distance metrics (L1, L2, L∞ norms).

#### `__main__`

The entry point of the script. Calls either `runComparisonModels()` or `testDifferentNorms()` based on the uncommented function.

---

## References

- **Scikit-learn PCA Documentation:** [Link](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- **Scipy KDTree Documentation:** [Link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html)
- **Multivariate Normal Distribution:** [Link](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)

---

**Note:** This project is for educational purposes to compare classification algorithms on the MNIST dataset. Ensure that you have the rights to use the dataset and comply with any licensing requirements.