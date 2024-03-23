'''
5	a. Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an appropriate data set for building the decision tree and apply this knowledge to classify a new sample.
'''

import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, label=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.label = label

def entropy(data):
    labels = data[:, -1]
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def split_data(data, feature_index, value):
    mask = data[:, feature_index] == value
    return data[mask]

def information_gain(data, feature_index):
    unique_values = np.unique(data[:, feature_index])
    split_entropy = 0
    for value in unique_values:
        subset = split_data(data, feature_index, value)
        probability = len(subset) / len(data)
        split_entropy += probability * entropy(subset)
    return entropy(data) - split_entropy

def find_best_split(data):
    num_features = data.shape[1] - 1
    best_feature = None
    best_gain = -1
    for i in range(num_features):
        gain = information_gain(data, i)
        if gain > best_gain:
            best_gain = gain
            best_feature = i
    return best_feature

def majority_label(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    index = np.argmax(counts)
    return unique_labels[index]

def id3(data):
    labels = data[:, -1]
    if len(np.unique(labels)) == 1:
        return Node(label=labels[0])
    if data.shape[1] == 1:
        return Node(label=majority_label(labels))
    best_feature = find_best_split(data)
    node = Node(feature=best_feature)
    unique_values = np.unique(data[:, best_feature])
    for value in unique_values:
        subset = split_data(data, best_feature, value)
        if len(subset) == 0:
            node.label = majority_label(labels)
            return node
        else:
            child_node = id3(subset)
            if node.value is not None and value == node.value:
                node.left = child_node
            else:
                node.right = child_node
    node.value = unique_values[0]  # Set the default value if the node has no value yet
    return node

def predict(node, sample):
    if node.label is not None:
        return node.label
    if sample[node.feature] == node.value:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)

def print_tree(node, depth=0):
    if node is None:
        return
    if node.label is not None:
        print(depth * '  ' + 'Predict:', node.label)
    else:
        print(depth * '  ' + 'Feature:', node.feature, 'Value:', node.value)
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

# Load data
data = pd.read_csv('tennisdata.csv')
X = data.drop('Play Tennis', axis=1).values
y = data['Play Tennis'].values

# Convert categorical data to numerical
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for i in range(X.shape[1]):
    label_encoders[i] = LabelEncoder()
    X[:, i] = label_encoders[i].fit_transform(X[:, i])

# Train decision tree
tree = id3(np.column_stack((X, y)))

# Print decision tree
print("Decision Tree:")
print_tree(tree)

# Test classification for a new sample
new_sample = ['Sunny', 'Cool', 'Normal', 'Weak']
for i in range(len(new_sample)):
    new_sample[i] = label_encoders[i].transform([new_sample[i]])[0]

prediction = predict(tree, new_sample)
print("Predicted class label:", prediction)
