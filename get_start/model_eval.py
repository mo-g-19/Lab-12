#4
#Copied from scikit-learn getting_started
#https://scikit-learn.org/stable/getting_started.html

"""Fitting a model -> data not mean predict well unseen data
Needs directly evaluated. Seen on train_test_split helper (used in pipelines.py)
Other tools for model eval (particular: cross-validation)
"""

#Below -> 5-fold cross-validation procedure (use cross_validate helper)
#   possible manually iterate over folds (use different data splitting strategies + custom scoring functions)
#   User Guide: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()

result = cross_validate(lr, X, y)   #defaults to 5-fold CV
print(result['test_score'])    #r_squared score is high because dataset is easy