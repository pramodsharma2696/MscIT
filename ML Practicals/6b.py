'''
6b Implement the classification model using clustering for the following techniques with K means clustering with Prediction, Test Score and Confusion Matrix.
'''
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit K-means clustering on training data
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_train)

# Assign labels to clusters
cluster_labels = kmeans.labels_

# Determine the majority class in each cluster
cluster_majority_class = []
for i in range(k):
    cluster_indices = np.where(cluster_labels == i)[0]
    majority_class = np.bincount(y_train[cluster_indices]).argmax()
    cluster_majority_class.append(majority_class)

# Predict the labels for test data
y_pred = []
for data_point in X_test:
    # Find the nearest cluster centroid
    nearest_cluster = kmeans.predict([data_point])[0]
    # Assign the label of the nearest cluster
    y_pred.append(cluster_majority_class[nearest_cluster])

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
