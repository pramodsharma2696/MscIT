'''

10A . Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an appropriate data set for building the decision tree and apply this knowledge to classify a new sample.

'''

from collections import Counter
import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, value=None, label=None):
        self.feature = feature  # Feature to split on
        self.value = value      # Value of the parent feature to reach this node
        self.label = label      # Predicted label if this is a leaf node
        self.children = {}      # Dictionary to store children nodes

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X, y, feature):
    pivot = np.mean(X[:, feature])
    y_left = y[X[:, feature] <= pivot]
    y_right = y[X[:, feature] > pivot]
    H_parent = entropy(y)
    H_children = (len(y_left) / len(y)) * entropy(y_left) + (len(y_right) / len(y)) * entropy(y_right)
    return H_parent - H_children

def id3(X, y, features):
    if len(set(y)) == 1:  # If all labels are the same, return a leaf node with that label
        return Node(label=y[0])
    if len(features) == 0:  # If there are no more features to split on, return a leaf node with the majority label
        return Node(label=Counter(y).most_common(1)[0][0])

    # Choose the best feature to split on based on maximum information gain
    best_feature = max(features, key=lambda feature: information_gain(X, y, feature))

    # Remove the chosen feature from the list of features
    remaining_features = [f for f in features if f != best_feature]

    # Create a new internal node to split on the best feature
    node = Node(feature=best_feature)

    # Recursively create child nodes for each unique value of the chosen feature
    for value in np.unique(X[:, best_feature]):
        X_subset = X[X[:, best_feature] == value]
        y_subset = y[X[:, best_feature] == value]
        node.children[value] = id3(X_subset, y_subset, remaining_features)

    return node

def predict(node, sample):
    if node.label is not None:  # If the current node is a leaf node, return its label
        return node.label
    if sample[node.feature] in node.children:  # If the feature value exists in the children, recursively call predict
        return predict(node.children[sample[node.feature]], sample)
    else:  # If feature value is not in children, return the label of the majority class at this node
        child_labels = [predict(child, sample) for child in node.children.values()]
        return Counter(child_labels).most_common(1)[0][0]


# Load the Iris dataset
iris = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
X = iris.drop(columns=['species']).values
y = iris['species'].values

# Split dataset into training and testing sets
X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]

# Train the decision tree using ID3 algorithm
features = list(range(X_train.shape[1]))  # Features to consider for splitting
decision_tree = id3(X_train, y_train, features)

# Make predictions on the testing set
y_pred = [predict(decision_tree, sample) for sample in X_test]

# Calculate accuracy
accuracy = sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)
