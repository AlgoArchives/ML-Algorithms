## Decision Tree

### Decision Tree from Scratch:

1.  **Data Preparation:**
    
    *   The code begins by defining a hardcoded CSV-like dataset representing weather conditions and whether to play tennis.
    *   Pandas is used to convert this data into a DataFrame and then into a list of lists.
2.  **Node Class:**
    
    *   A `Node` class is defined to represent nodes in the decision tree.
    *   Each node can have an attribute, value, results (if it's a leaf node), true branch, and false branch.
3.  **Utility Functions:**
    
    *   `unique_counts`: Calculates the frequency of each class label in a set of rows.
    *   `entropy`: Calculates the entropy of a set of rows based on class label frequencies.
    *   `divide_data`: Splits a set of rows based on a given column and value.
4.  **Tree Building:**
    
    *   `build_tree` function recursively builds the decision tree.
    *   It calculates the best split (attribute and value) using information gain and divides the data into true and false branches.
    *   The process continues until a stopping condition is met (e.g., maximum depth or minimum samples per leaf).
5.  **Tree Visualization:**
    
    *   `print_tree` function recursively prints the decision tree structure.

### Decision Tree using scikit-learn:

1.  **Data Loading and Splitting:**
    
    *   The code uses scikit-learn's `load_iris` function to load the Iris dataset.
    *   The dataset is split into training and testing sets using `train_test_split`.
2.  **Model Training and Evaluation:**
    
    *   A decision tree classifier from scikit-learn is instantiated and trained on the training data.
    *   Predictions are made on the testing data, and accuracy is calculated using `accuracy_score`.

### Explanation:

*   The decision tree from scratch code implements the core concepts of decision tree learning, including entropy calculation, information gain, recursive tree building, and tree visualization.
*   It's a simplified version and may not handle all scenarios or optimizations present in library implementations like scikit-learn's decision tree.
*   The scikit-learn code demonstrates how to use a pre-built and optimized decision tree classifier with minimal code, suitable for practical machine learning tasks.
*   The choice between building a decision tree from scratch or using a library depends on factors like complexity, performance requirements, and available resources.