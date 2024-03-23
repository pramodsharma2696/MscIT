#1.a.Design a simple machine learning model to train the training instances and test the same.


# 1. Load the dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Choose a simple machine learning model
from sklearn.tree import DecisionTreeClassifier

# 5. Train the model on the training set
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 6. Test the model on the testing set
y_pred = model.predict(X_test)

# 7. Evaluate the model's performance
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
