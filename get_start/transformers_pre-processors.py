#2
#Copied from scikit-learn getting_started
#https://scikit-learn.org/stable/getting_started.html

from sklearn.preprocessing import StandardScaler

"""ML workflows composed differ parts. Typical pipeline consist
pre-processing stemp transform or impute data (final prdictor 
predicts target values)"""

#In here, pre-processors and transformers follow same API as estimator
#objects (actually all inherit from same BaseEstimator)
#transformer objects not have predict method (transform method
#outputs newly transformed smple matrix X:)

from sklearn.preprocessing import StandardScaler

X = [[0, 15],
    [1, -10]]

#scale data according to compute scaling values
print(StandardScaler().fit(X).transform(X))
