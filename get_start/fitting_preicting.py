#1
#Copied from scikit-learn getting_started
#https://scikit-learn.org/stable/getting_started.html

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)

"""Demonstrating the fit method.
There are dozens of built-in ML algorithms and models (estimators), and each estimator = fit to data (fit funct)"""

#Fit takes 2 in (sample matrix (X) and target values (y))
#   X = typically (n_samples, n_features)
#   y = usually 1d array (i entry correspond target of ith sample (row) of X)
#       real numbers for regression tasks (or integers for classification)
#X and y => numpy arrays or equivalent array-like (some estimators => sparse matricies)

#2 smples, 3 features
X = [[1, 2, 3],
    [11, 12, 13]]

#classes of each sample
y = [0, 1]

clf.fit(X, y)

#Since fit => not need re-train estimator
print(clf.predict(X))
#Can used predict target new data
print(clf.predict([[4, 5, 6], [14, 15, 16]]))

#Choosing the right estimator: https://scikit-learn.org/stable/machine_learning_map.html#ml-map
"""
Flow chart on choosing right estimator based on data that have
"""