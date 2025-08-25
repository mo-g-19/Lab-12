#This is my comprehensive notes from https://www.geeksforgeeks.org/machine-learning/comprehensive-guide-to-classification-models-in-scikit-learn/
#Not my code

#In class, learned there are multiple types of models, classification, regression, clustering, reduce dimension, deep learning, and more
#I recognize Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines, Decision Trees, and Random Forest
#I'll go over it, but try to not spend too much time on it

"""What is classification? - supervised learning tech => goal predict category class labels of new instances based past observations.
        Involves: train a model on labeled dataset, target vaiable is categorical

Key Concepts
        Features: input var used make predictions
        Label: output var model try predict
        Training data: dataset used train model
        Test Data: dataset used eval performance"""

"""Scikit-Learn Classification Models
Logistic Regression - used binary classification problems. Model probability given imput belong particular class
    Adv:
        Simplicity and Interpretability: easy implement + interpret. Clear probabilistic framework binary classify
        Efficiency: computationally efficient + well large datasets
        No Need Feature Scaling: Logistic regression not requrie feature scaling (simpler use)
    Dis:
        Linear Decision Boundary: assume linear relationship between independent variables and log-odd
            dependet var, *may not true
        Overfitting with High-Dimensional Data: number of observations < number for features
        Not suitable for non-linear problems: not handle non-linear relationships unless fetures transformed

Implementation -> relationship between input features + probability binary outcome use logistic funct (sigmoid funct)"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report