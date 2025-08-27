#Authors: scikit-learn devs
#SPDX-License-Identifier: BSD-3-Clause

#Standad scientific Python imports
import matplotlib.pyplot as plt

#Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

#If working from image ('png') -> matplotlib.pyplot.imread library
#8x8 grayscale values for each image (use first 4 digits)

"""Data loading -> used the load_* so either from database or just lots of storage locally"""
digits = datasets.load_digits()

"""I guess this is pre-processing. Showing the image for first 4 digits, and showing them what the 'target value' (correct digit) is"""
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.taget):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

"""In documentation of classification:
        Flaten 8x8 into 1D 64
        Dataset (n_samples, n_features); samples -> num images + features -> total num pixels each image

    Split to train + test subset -> support vector classifier
        https://medium.com/@swetaprabha/support-vector-classifier-and-support-vector-machine-4a675b8cac88
            "soft margin classifiers and intentionally misclassifies a few training observations to prevent overfitting"
                similiar optimization + slight changes; error term introduced + constrained non-neg tuning parameter (C)
                C -> either more tolerant margin violation (simpler) or leading overfitting (lower bias + increase variance)
            *Counters the common disadvantage of SVMs -> overfitting"""

"""I didn't necissarily select the training model, but this is choosing a model and then creating a model.
It also involves preprocessing data because the images are transformed into a 1D vector"""
#flatten images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

#Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

#Learn the digits on train subset
clf.fit(X_train, y_train)

#Predict value of digit on the test subset
predicted = clf.predict(X_test)

"""Visualize the first 4 test samples + predicted digit value in title
This is part of evaluation. Checking the results, visually. Then goes into classification report"""
#Previously compressed; now expanded back into image
_, axes = plt.subplots(nrows=1, ncols=4, figsize(10,3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

#Report
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

#The output doesn't have a configuration matrix
#I recognize precision, recall, and f1-score in the correct spot


#Accuracy is a row because it is the total correct / total (doesn't seperate by class)
#As seen in the example, the prediction only gets 3 out of 5 of the examples correct -> 0.60

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
#support is the sample size
#macro avg is just the straight mean of each evaluation; weighted avg is the weighted mean based on the sample size