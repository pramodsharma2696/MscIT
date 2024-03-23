'''
3.b. Write a program to implement Decision Tree and Random forest with Prediction, Test Score and Confusion Matrix.

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data from CSV
data = pd.read_csv('tennisdata.csv')

# Obtain train data and train output
X = data.drop(columns=['Play Tennis'])
y = data['Play Tennis']

# Convert categorical variables into numbers
X = pd.get_dummies(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Prediction
y_pred_dt = dt_classifier.predict(X_test)
y_pred_rf = rf_classifier.predict(X_test)

# Test score
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Decision Tree Test Accuracy:", accuracy_dt)
print("Random Forest Test Accuracy:", accuracy_rf)

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nDecision Tree Confusion Matrix:")
print(cm_dt)
print("\nRandom Forest Confusion Matrix:")
print(cm_rf)
