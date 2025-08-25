#3
#Copied from scikit-learn getting_started
#https://scikit-learn.org/stable/getting_started.html

""" Transformers and estimators can be combo together => object
Pipeline offers same API as regular estimator (can use fit and predict functs)
*Using pipeline => prevent data leakage (disclose some test data in train data)
"""

#Load Iris dataset (split into train and test sets), compute accuracy score of a pipeline on test data:
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

#load the iris dataset and split it into train and test sets
X, y = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#fit the whole pipeline
print(pipe.fit(X_train, y_train))

#can now use like any other estimator
print(accuracy_score(pipe.predict(X_test), y_test))
