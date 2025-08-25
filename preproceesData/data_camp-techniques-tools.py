#Taking notes from https://www.datacamp.com/blog/data-preprocessing
#Common Techniques and Tools using Python Libraries

"""Common Techniques for Data Preprocessing with examples"""

"""Handling missing data
Several strategies:
    Imputation: filling in missing values with calculated estimate (mean, media, or mode).
        Adv methods => predictive modeling (missing predicted based relationship within data)"""
#Dummy code + not run on own
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean') #Replace with 'mean'/'median'/'most_frequent'
data['column_with_missing'] = imputer.fit_transform(data[['column_with_missing']])

"""
    Deletion: Removing rows or columns with missing values
        Used cautitonsly => loss valuable data"""
#Dummy code
data.dropna(inplace=True)   #removes rows any missing val

"""
    Modeling missing values: cases missing data pattern more complex, ML models predict
        missing val based rest dataset (improve accuracy by incorporating relationships between diff var)"""

"""
Outlier detection and removal
    Z-Score method: measures how many standard deviations data point is from mean.
        Data points beyond threshold (Ex: +-3 standard deviation) => outliers"""
#Dummy -> not work unless data column named 'column imported

from scipy import stats
z_scores = stats.zscore(data['column'])
outliers = abs(z_scores) > 3 #Id outliers

"""
    Interquartile range (IQR): between Q1 and Q3 (val beyond 1.5 times IQR above Q3 or below Q1 => out)"""
#Dummy
Q1 = data['column'].quantile(0.25)
Q3 = data['column'].quantile(0.75)

IQR = Q3 - Q1 
outliers = (data['column'] < (Q1 - 1.5* IQR)) | (data['column'] > (Q3 + 1.5*IQR))

"""
    Visual techniques: box plots, scatter plots, or histograms. ID => remove or transformed"""

"""
Data encoding - convert categories into numerica representations
    One-hot encoding: method create binary col for each category"""
#Ex
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(data[['categorical_column']])

"""
    Label encoding: assing unique numerical val each category
        *Can introduce unintended ordinal relationship between categories if no natural order"""
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
data['ordinal_column'] = oe.fit_transform(data[['ordinal_column']])

"""
Data scaling and normalization - scaling and normalization -> similiar scale (important kNN and SVMs)
    Min-max scaling: data sepecficed range (typic 0-1).
        Useful all features need have same scale"""
#Ex
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['scaled_column']] = scaler.fit_transform(data[['numeric_column']])

"""
    Standardization (Z-score normalization): scales data that mean -> 0 and standard deviation = 1
        Help model perform better with normally distributed features"""
#Ex
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['staardized_column']] = scaler.fit_tranform(data[['numeric_column']])

#Difference between Standardization and Normalization:
#   Standardization: gen = adjust feature values by subtacting mean + deviding by standard deviation ('centering and scaling')
#       Centering: Xstd (standardized value) = ((X (og) - mu (mean)) / sigma (standard deviation of feature))
#   Normalization: gen = process adjust values -> common scale on specific range

"""
Data augmentation - artificially increase size dataset -> create new, synthetic examples.
Usefully image or text dataset in Deep Learning models (large amounts data required robust model performance)
    Image aumentation: rotating, flipping, scaling, or adding noise images -> variations improve model generalization"""
#Ex
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

augmented_images = datagen.flow_from_director('image_dircetory', target_size=(150, 150))

"""
    Text augmentation: include synonym replacement, random insertion, and back-translation where
        a sentence is translated into another language and then back to original (include variations)"""
#Example
import nlpaug.augmenter.word as naw
#install nlpaug here: https://github.com/makcedward/nlpaug

aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = aug.augment("This is a sample text for augmentation.")

"""Tools for Data Preprocessing
Can implement using pure Python, powerful tools => more efficient.
    Pandas: most common for data manipulation and cleaning. Flexible data structures
        primary DataFrame and Serie, enable handle and manipulate structured data efficiently.
        Operations handling missing data, merging datasets, filtering data, and reshaping"""
import pandas as pd

#Load a sample dataset
data = pd.DataFrame({
    'name': ['John', 'Jane', 'Jack'],
    'age': [28,31,34]
})

print(data)

"""
    NumPy: fundamental library for numerical computations.
        Support large, multi-dimensional arrays and matrices and math functions operate on arrays.
        Foundation many higher-level data process libraries (ex: Pandas)"""
import numpy as np

#Create an array and perform element-wise operaions
array = np.array([1, 2, 3, 4])
squared_array = np.square(array)

print(squared_array)

"""
    Scikit-learn: widley used ML tasks + numerous preprocessing utilities
        (scaling, encoding, and data transformation). Preprocessing module
        tools handling categorical data, scaling numerical data, feature extraction, etc"""
from sklearn.preprocessing import StandardScaler

#Standardize data
data = [[10, 2], [15, 5], [20, 3]]
scaler = StandardScaler()
scaled_data = scaler.fit_tranform(data)
print(scaled_data)

"""
    Cloud platforms -> op-premise system not handle large datasets effectively -> vast amounts data across distributed systems
        AWS Glue: fully managed ETL by Amazon Web. Auto discovers and organizes
            data and prepares analytics. Support data cataloging and connect AWS services
            like S3 and Redshift
        Azure Data Factory: cloud-based data integration from Microsoft. Support building
            ETL and ELT pipelines for large-scale data. Allows users move data
            between various services, preprocess using transformations, and orchestrate
            workflows use visual interface"""
"""
    Automation tools - repetative steps
        AutoML (Automated ML) platfoms: automate several stages ML workflow
            Google's AutoML, Microsoft Azure AutoML, and H20.ai's AutoML -> automated pipelines
            handle tasks like feature selection, data transformation, and model selection minimal user intervention
        Preprocessing pipelines in scikit-learn: Provides Pipeline class (help streamline + automate preprocessing steps).
            Allow string multiple preprocessing operations into single, executable workflow,
            ensuring preprocess tasks applied consistent"""
#Scikit-Learn Pipeline class example
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SingleImputer
from sklearn.compose import ColumnTransformer

#Example Pipeline combining different preprocessing tasks
numeric_transformer = Pipeline(steps[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, ['age']),
    ('cat', categorical_transformer, ['category'])
])

preprocessed_data = preprocessor.fit_tranform(data)