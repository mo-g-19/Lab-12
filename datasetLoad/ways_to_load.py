#Not a trustworthy source, but a quick 10 minute explaination of 5 different ways to upload could be useful
#Work is from https://blog.finxter.com/5-best-ways-to-load-data-using-the-scikit-learn-library-in-python/

""" Method 1: load_* funct => Built-in Datasets
Several built-in datasets accessible through functions starting wth load_*
Datasets useful experimenting algorithms and loaded as Bunch object
    (like dictionaries => data, target, and descriptive keys)
"""
#Same example in the Getting Started Guide fro Scikit-Learn
"""from sklearn.datasets import load_iris
iris_data = load_iris()
X, y = iris_data.data, iris_data.target
"""


"""Method 2: fetch_* => loading from External Datasets
Larger datasets/not inlcluded, fetch_* => retrieve data from internet
    Also return Bunch objects + ideal working real-world data
"""
#Accesses the MNIST dataset, large collection handwrtten digits => train image process systems
#   Obtain data + target arrays ready preprocessing + model train
"""from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target
"""


"""Method 3: load_svmlight_file => Importing for Sparse Data
tailored loading datsets in SVMlight format (adv sparse datasets).
Returns data and target suitable feeding directly estimator's fit() *fit funct mentioned get_start*
"""
#Have a large and sparse dataset (SVM classifiers), read data + unnecessary increase mem usage (keep sparse format)
"""from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file('my_datset.txt')
"""
#>>Out: sparse matrix (X) and array target values (y)


"""Method 4: load_files => Loading Text Files
loading txt organized into folders by class (common text classification tasks)
    Returns Bunch object encapsulating training data and labels
"""
#Particularly useful natural language processing task (category directory per class)
#   Convinent load + vectorize textual data for classification alg
"""from sklearn.datasets import load_files
text_data = load_files('txt_dataset/')
X, y = text_data.data, text_data.target
"""


"""*Bonus* Method 5: pandas.read_csbv => One-Liner load CSV
not strictly scikit-learn funct, effective one-liner load CSV -> used scikit-learn ML models
"""
#Importing CSV file into DataFrame, acquire versatile data structure ready preprocessing, exploration, + feeding into scikit models
"""import pandas as pd
df = pd.rea_csv('my_data.csv')
"""