'''
7a Implement the classification model using clustering for the following techniques with hierarchical clustering with Prediction, Test Score and Confusion Matrix
'''
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Define static data within the code
data = pd.DataFrame({
    "feature1": [10, 25, 15, 18, 40, 30],
    "feature2": [5, 8, 10, 12, 6, 9],
    "target_label": ["A", "B", "A", "B", "C", "A"]
})

# Separate features and target labels
X = data.drop("target_label", axis=1)
y = data["target_label"]

# Optional: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical clustering (adjust the number of clusters as needed)
hier_clustering = AgglomerativeClustering(n_clusters=2)
cluster_labels = hier_clustering.fit_predict(X_scaled)

# Combine engineered features or use cluster labels directly (adapt for your approach)
combined_features = X_scaled  # Adjust based on your approach

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(combined_features, y, test_size=0.2)

# Classification model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
test_score = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

print("Test Score:", test_score)
print("Confusion Matrix:\n", confusion_matrix)
