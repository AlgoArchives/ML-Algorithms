Gradient Boosting Regression
====================================

Gradient boosting is a machine learning technique used for regression and classification tasks. It works by combining multiple weak learners (typically decision trees) to create a strong predictive model.

The code examples in this repository demonstrate how to:
--------------------------------------------------------

*   Perform gradient boosting regression on the Boston housing dataset using scikit-learn.
*   Load a CSV file with hardcoded data and apply gradient boosting regression.

Usage
-----

1.  **Boston Housing Dataset Example:**
    
    *   Run the `boston_gradient_boost.py` script to perform gradient boosting regression on the Boston housing dataset.
    *   This script fetches the Boston housing dataset from its original source, splits the data, trains the model, and evaluates its performance.
2.  **Hardcoded Data Example:**
    
    *   Run the `hardcoded_gradient_boost.py` script to perform gradient boosting regression on hardcoded data from a CSV file.
    *   This script creates a CSV file with hardcoded data, loads the data, splits it, trains the model, and evaluates its performance.

Files
-----

*   `boston-boost.py`: Python script for gradient boosting regression on the Boston housing dataset.
*   `more-info-boston-boost.py`: Python script for gradient boosting regression on the Boston housing dataset with more information with each stump's data.
*   `hardcoded-data-boost.py`: Python script for gradient boosting regression on hardcoded data from a CSV file.
*   `hardcoded_data.csv`: CSV file containing hardcoded data for the second example.