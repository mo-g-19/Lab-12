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
#flatten images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))