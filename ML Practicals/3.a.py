'''
3.a Write a program to implement the naïve Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.

'''

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV
data = pd.read_csv('tennisdata.csv')
print('The first five values of data are:\n', data.head())

# Obtain train data and train output
X = data.drop(columns=['Play Tennis'])
print('\nThe first five values of train data are:\n', X.head())

y = data['Play Tennis']
print('\nThe first five values of train output are:\n', y.head())

# Convert categorical variables into numbers
le_outlook = LabelEncoder()
X['Outlook'] = le_outlook.fit_transform(X['Outlook'])
le_temperature = LabelEncoder()
X['Temperature'] = le_temperature.fit_transform(X['Temperature'])
le_humidity = LabelEncoder()
X['Humidity'] = le_humidity.fit_transform(X['Humidity'])
le_wind = LabelEncoder()
X['Wind'] = le_wind.fit_transform(X['Wind'])

print('\nNow the train data is:\n', X.head())

le_play_tennis = LabelEncoder()
y = le_play_tennis.fit_transform(y)
print('\nNow the train output is:\n', y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
