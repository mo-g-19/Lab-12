#This is my comprehensive notes from https://www.geeksforgeeks.org/machine-learning/comprehensive-guide-to-classification-models-in-scikit-learn/
#Not my code

##ALL OUTPUT VALUES FROM THE SITE ARE BASED ON Load_wine dataset and a specific random state for splitting the data

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

X, y = load_wine(return_X_y=True)
X_train, X_test, y_tain, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#>>Output - Accuracy value
#   A table with col precision, recall, f1-score, and support to 3 different sets
#   An additional table for accuracy, macro avg, and weighted avg

"""
        When to use:
            Ideal: binary classification problems where featues + target var approximately linear
            Useful: baseline model -> simplicity and target var"""


"""
K-Nearest Neighbors (KNN) - simple, instance-based learning alg. Classifies data point based on majority class among k-NN
        Adv:
            No training period: lazy learner (no train phase) => fast implement
            Flexibility: handle multi-class classification prolems + easy understand and implement
            Adaptability: New data can added seamlessly without retrain model
        Dis:
            Computationally Intensive: can slow large datasests (requires computing distance new point and all exist point)
            Sensitive to Noise and Outliers: sensitive => affect performance
            Feature Scaling requried: ensure all features contribut equally to distance calcs
Implementing -> calc distance new data points and all existing points, assigns class of majority of kNN"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_ped_knn = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

#>>Output - Accuracy
#   Same format as the table in Logistic Regression with similar values
#   All the following tables have the same col and row values (including measurements)

"""
        When to Use:
            Suitable: small - medium size datasets -> computational cost manageable
            Useful: problems decision boundary not linear and can be complex"""

"""
Support Vector Machines (SVM) - powerful classifier -> find hyperplane best seperate classes in feature space
        Adv:
            Effective in High-Dimensional Spaces: well with high-dimensional data
                + effective number dimensions exceed number samples
            Robust to Overfitting: good generalizaition performance + less prone overfitting
                especially use regularization
            Versatility: Handle both linear and non-linear classification using kernal trick
        Dis:
            Computationally Expenive: slow to train (especially large datasets)
            Choice of Kernel" select papropriate kernal and tuning hyperparameters challenge
            Interpretability: SVM less interpretable compared simpler models (logistic regression)
Implementig - SVM find hyperfplane max the margin between closest points differ classes
    (non-linear) -> kernel functions transform data to higher dimensions"""
from sklearn.svm import SVC 

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
#>>Out: look at the site

"""
        When to use:
            Ideal: complex classification high-dimension data + decision boundary non-linear
            Useful: dataset small + many features"""

"""
Decision Trees - non-parametric models split data subsets based value input features. Easy interpret + visualize
        Adv:
            Interpretability: Decision Trees easy understand + interpret, make useful
                explain model predictions to non-technical
            Handle Different Data Types: both numberical and categorical without req feature scaling or encoding
            Non-Linear Relationships: caputre non-lin between features and target var
        Dis:
            Prone Overfitting: easily overfit train data (especially if deep + complex)
            High Variance: Small changes data -> significantly different tree (unstable)
            Bias To Dominant Classes: toward features many levels/dominant classes in imbalanced datasets
Implementing Decision Trees - alg split data subsets based value input features (create branches til reach decision node (leaf))"""
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5) #I remember seeing this previously (only could find in auto_param_search.py)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
#>>Output: on site

"""
        When to use:
            Suitable: where interpretability is crucial + relationships features <=> target var non-linear
            Useful: quick data exploration + feature selection"""

"""
Random Forest - ensemble method combines multiple decision trees imporve model's accuracy + robustness
        Adv:
            High accuracy: random forest generally provide high accuracy
                averaging predictions multiple decision trees (reduce variance)
            Robustness to Noise: relilient to noisy data + outliers, suitable real-world datasets
            Feature Importance: Provide insight feature importance (aid feature selection + understand dataset)
        Dis:
            Computationally Intensive: Build multiple decison trees -> expensive and more resources
            Interpretability: Ensemble nature challenge interpret reasoning behind individual predictions compared single tree
            Bias in Imbalanced Datasets: Random forest biased toward majority class in imbalanced datasets, affect performance for minority class
Implementing - multiple decision trees use random subsets of data + features, average predictions"""
from sklearn.ensemble import RandomForestClassifier #Remember using specifically the Random Forest Classifier before *This is was in the User Guide (specifically fitting + predicting and automatic parameter searches)

forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
forrest.fit(X_train, y_train)

y_pred_forest = forest.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_forest))
print(classification_report(y_test, y_pred_forest))
#>>Output: on site
"""
        When to use:
            Ideal: complex classification problems large + high-dimensional datasets
            Useful: robustness to noise + feature importance insights requried"""

"""
Naive Bayse - based Bayes' theorum and assume independence between features (effective text classification)
        Adv:
            Well with High-Dimensional Data: such as text classification tasks
            Handles Multi-Class Problems: effective multi-class prediction problems
            Less Training Data Required: well with less training data if independence assumption holds
        Dis:
            Independence Assumption: assumes all features indpendent (rarely true irl). limit accurace when correlated
            Zero-Frequency Problem: if categorical var in test data has category not present train,
                assign it 0 probability (mitigated use smoothing techniques)
            Poor Estimation: probability outputs form predict_proba not taken too seriously
            Sensitivity to Irrelevant Features: can be sensitive presence irrelevant features (affect performance)
Implementing - assumption of independence between features (calcs probability each class given input features + select class highest prob)"""
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
#>>Out: on site
"""
        When to use:
            Suitable: (speed) applications for real-time predict
            Useful: problems involve multiple classes"""

"""
Gradient Boosting - ensemble technique builds models sequentiall, each correct error predecessor
        Adv:
            High Predictive Accuracy: often provide predictive accuracy difficult surpass
            Flexibility: optimize different loss functions and offer several hyperparameter tuning options (high flexibile)
            No Data Pre-Processing Requried: often well both categorical + numerical val no extensive data pre-process
            Handle Missing Data: without need imputation
        Dis:
            Overfitting: Gradient Boosting mod can overfit train (especially not properly regularized).
                Mitigated techniques penalized learning, tree constraints, randomized sampling, and shrinkage
            Computationally Expensive: often requires many trees (sometimes more than 1000), can time and memory exhaustive
            Complex Hyperparameter Tuning: high flexibility results many parameters interact + influence model behavior,
                require extensive grid search during tuning
Implementing - combines weak learners (typically decision trees) in sequential manner (each new model focus residual error previous)"""
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
#>>Out: on site
"""
        When to use:
            Good choice: dataset missing values
            Suitable: relationships featuresand target = complex + non-linear"""


"""Quick on Evaluation Metrics (eval = crucial + Scikit-Learn = several metrics asses model perform)
        Accuarcy: ratio correctly predicted to total
        Precision: ratio correctly predicted positive observations to total predicted positives
        Recall (Sensitivity): ratio correctly predicted positive observations to all observations in actual class
        F1-Score: weighted average of Precision and Recall
        Confusion Maxtrix: table used describe performance classification model"""
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

precision = precision_score(y_test, y_pred, acerage='weighted')
print("Precision:", precison)

recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)

f1 = f1_score(y_test, y_pred, average='weighted')
print("F1-Score:", f1)
#>>Out: on site

"""Practical Tips using Scikit-Learn
    Data Preprocessing: always preprocess data (missing val, scaling features, encoding categorical var)
    Feature Selection: use techniques like Recursive Feature Elimination (RFE) -> most relevant features
    Cross-Validation: Use cross-validation get more accurate estimate model performance
    Model Interpretation: tools like SHAP and LIME to interpret model prediction"""