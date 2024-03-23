'''
2.a.Perform Data Loading, Feature selection (Principal Component analysis) and Feature Scoring and Ranking

Install:

pip install scikit-learn
pip install pandas

'''

from pandas import read_csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler

# Step 1: Data Loading
def load_data(file_path, names):
    return read_csv(file_path, names=names)

# Step 2: Feature Selection using Univariate Statistical Test (Chi-squared)
def perform_chi_squared_feature_selection(X, Y, k):
    test = SelectKBest(score_func=chi2, k=k)
    fit = test.fit(X, Y)
    return fit

# Step 3: Feature Scoring and Ranking
def score_and_rank_features(selected_features, names):
    selected_indices = selected_features.get_support(indices=True)
    selected_features_names = [names[i] for i in selected_indices]
    return selected_features_names

def main():
    # Step 1: Data Loading
    filename = 'diabetes.csv'
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    dataframe = load_data(filename,names)

    # Extract features and target variable
    array = dataframe.values
    X = array[:,0:8]
    Y = array[:,8]

    # Step 2: Feature Selection using Univariate Statistical Test (Chi-squared)
    k = 4  # Select the top 4 features
    selected_features = perform_chi_squared_feature_selection(X, Y, k)

    # Step 3: Feature Scoring and Ranking
    selected_features_names = score_and_rank_features(selected_features, names)

    # Summarize selected features
    print("Selected features:", selected_features_names)

if __name__ == "__main__":
    main()
