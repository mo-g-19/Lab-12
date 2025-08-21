#Example from https://www.tutorialspoint.com/how-can-scikit-learn-library-be-used-to-load-data-in-python

#load_iris seems to be a common dataset
from sklearn.datasets import load_iris

my_data = load_iris()
X = my_data.data
y = my_data.target

feature_name = my_data.feature_names
target_name = my_data.target_names

print("Feature names are : ", feature_name)
print("Target names are : ", target_name)
print("\nFirst 8 rows of the dataset are : \n", X[:8])