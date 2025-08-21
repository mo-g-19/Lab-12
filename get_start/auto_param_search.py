#5
#Copied from scikit-learn getting_started
#https://scikit-learn.org/stable/getting_started.html

"""All estimators have parameters (often hyper-parameters) can tuned.
Generalizing power of estimator often critically depend few parameter
    Ex: RandomForestRegressor = n_estimtors (determine num of trees in forest)
        and max_depth (determine max depth of each tree)
    Often, not clear exact values param should be since depend on data

Scikit-learn = tools auto find best parameter combo (via cross-validation)
"""

#randomly search over parameter space random forrest (RandomizedSearchCV object)
#search over, RandomizedSearchCV behaves as RandomForestRegressor fitted best set of parameters
#   User Guide: https://scikit-learn.org/stable/modules/grid_search.html#grid-search

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint

#Create a synthetic dataset
X, y = make_regression(n_samples = 20640,
                        n_features=8,
                        noise=0.1,
                        random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#define the parameter space that will be searched over
param_distributions = {'n_estimators' : randint(1, 5),
                        'max_depth': randint(5,10)}

#now create a searchCV object and fit it to the data
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                            n_iter=5,
                            param_distributions=param_distributions,
                            random_state=0)
print(search.fit(X_train, y_train))
print()
print(search.best_params_)
print()

#the search object now act likenormal random forest estimator
#with max_depth=9 and n_estimators=4
print(search.score(X_test, y_test))