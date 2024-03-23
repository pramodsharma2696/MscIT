import pandas as pd

# Step 1: Data Loading
def load_data(file_path):
    return pd.read_csv(file_path)

# Step 2: Candidate-Elimination Algorithm
def candidate_elimination(data):
    n_attributes = len(data.columns) - 1
    specific_h = ['0'] * n_attributes
    general_h = ['?'] * n_attributes

    # Iterate over each row in the dataset
    for index, row in data.iterrows():
        if row.iloc[-1] == 'Yes':  # Positive example
            for i in range(n_attributes):
                if specific_h[i] == '0':
                    specific_h[i] = row[i]
                elif specific_h[i] != row[i]:
                    specific_h[i] = '?'
                    general_h[i] = '?'
        else:  # Negative example
            for i in range(n_attributes):
                if row[i] != specific_h[i]:
                    general_h[i] = specific_h[i]
                else:
                    general_h[i] = '?'
    
    return [specific_h, general_h]

# Step 3: Demonstration
def main():
    # Load the training data
    file_path = 'train.csv'  # Specify the path to your training data file
    training_data = load_data(file_path)
    
    # Apply the Candidate-Elimination algorithm
    hypotheses = candidate_elimination(training_data)
    
    # Print the set of hypotheses
    print("Specific Hypothesis:", hypotheses[0])
    print("General Hypothesis:", hypotheses[1])

if __name__ == "__main__":
    main()
