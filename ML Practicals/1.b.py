
'''
1.b. Implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a .CSV file	5
Source with output	5
Data Source TrainingData.csv


'''

import csv

def find_s_algorithm(training_data):
    # Initialize hypothesis with the first training instance
    hypothesis = training_data[0][:-1]

    # Iterate over each training instance
    for instance in training_data:
        # If the instance is positive, update the hypothesis
        if instance[-1] == 'Yes':
            for i in range(len(hypothesis)):
                if instance[i] != hypothesis[i]:
                    hypothesis[i] = '?'
    
    return hypothesis

def read_training_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        training_data = list(reader)
    
    return training_data

def main():
    # Read training data from CSV file
    training_data = read_training_data('TrainingData.csv')

    # Implement FIND-S algorithm
    hypothesis = find_s_algorithm(training_data)

    # Print the most specific hypothesis
    print("The most specific hypothesis is:", hypothesis)

if __name__ == "__main__":
    main()
