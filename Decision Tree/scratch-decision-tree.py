import pandas as pd

# Sample hardcoded CSV data
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

class Node:
    def __init__(self, attribute=None, value=None, results=None, true_branch=None, false_branch=None):
        self.attribute = attribute
        self.value = value
        self.results = results
        self.true_branch = true_branch
        self.false_branch = false_branch

def unique_counts(rows):
    results = {}
    for row in rows:
        label = row[-1]
        if label not in results:
            results[label] = 0
        results[label] += 1
    return results

def entropy(rows):
    from math import log
    log2 = lambda x: log(x) / log(2)
    results = unique_counts(rows)
    entropy = 0.0
    for label in results:
        p = float(results[label]) / len(rows)
        entropy -= p * log2(p)
    return entropy

def divide_data(rows, column, value):
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    true_rows = [row for row in rows if split_function(row)]
    false_rows = [row for row in rows if not split_function(row)]
    return true_rows, false_rows

def build_tree(rows):
    if len(rows) == 0:
        return Node()
    current_entropy = entropy(rows)
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(rows[0]) - 1
    for col in range(column_count):
        column_values = set([row[col] for row in rows])
        for value in column_values:
            true_rows, false_rows = divide_data(rows, col, value)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = current_entropy - (len(true_rows) / float(len(rows)) * entropy(true_rows) + len(false_rows) / float(len(rows)) * entropy(false_rows))
            if gain > best_gain:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (true_rows, false_rows)
    if best_gain > 0:
        true_branch = build_tree(best_sets[0])
        false_branch = build_tree(best_sets[1])
        return Node(attribute=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)
    else:
        return Node(results=unique_counts(rows))

def print_tree(node, spacing=""):
    if node.results is not None:  # Leaf node
        print(spacing + "Predict", node.results)
        return
    print(spacing + f"{df.columns[node.attribute]}: {df.columns[node.value]}?")
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

# Convert categorical data to numerical values
df['Outlook'] = df['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
df['Temperature'] = df['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
df['Humidity'] = df['Humidity'].map({'High': 0, 'Normal': 1})
df['Wind'] = df['Wind'].map({'Weak': 0, 'Strong': 1})
df['PlayTennis'] = df['PlayTennis'].map({'No': 0, 'Yes': 1})

# Convert dataframe to list of lists
data = df.values.tolist()

# Build the decision tree
tree = build_tree(data)

# Print the decision tree
print_tree(tree)