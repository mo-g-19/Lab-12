#Taking notes from https://www.datacamp.com/blog/data-preprocessing
#I'm putting Best Practices first because that's the bigger takeaway to remember

"""What is Data Preprocessing
    refers any processing applied raw data -> ready future analysis or processing tasks
    prelininary step in data analysis + recently -> train ML and AI models -> inferences
"""

"""Best Practices for Data Preprocessing
    Understand the data
        Conduct exploratory data anaylsis to identify strutucture data at hand.
            Specifically:
                Key features
                Potential anomalies
                Relationships
    Automate repetitive steps
        Often involves repetitive steps.
            Automate:
                build pipelines -> consistency + efficiency (reduce manual errors)
                Use scikit-learn or cloud-based
    Document preprocessing steps
        Two objectives: Reproducibility + Understanding (later date or others)
    Iterative improvements
        Should be iterative process => as models evolve + provie feedback performance => revisit + refine -> better results
            Instance, feature engineeing => new useful features or tuning outlier handling -> improve model accuracy
"""

"""Steps in Data Prepocessing
Step 1: Data cleaning
    Correct errors/inconsistencies => accurate and complete
        Ex:
            Handling missing values
            Removing duplicates
            Correcting inconsistent formats"""
import pandas as pd

#Creating a manual dataset
data = pd.DataFram({
    'name': ['John', 'Jane', 'Jack', 'John', None],
    'age': [28, 34, None, 28, 22],
    'purchase_amount': [100.5, None, 85.3, 100.5, 50.0],
    'date_of_purchase' : ['2023/12/01', '2023/12/02', '2023/12/01', '2023/12/01', '2023/12/03']
    })

#Handling missing values using mean imputation for 'age' and 'purchase_amount'
imputer = SimpleImputer(strategy='mean')
data[['age', 'purchase_amount']] = imputer.fit_transform(data[['age', 'purchase_amount']])

#Removing duplicate rows
data = data.drop_duplicates()

#Correcting inconsistent date formats
data['date_of_purchase'] = pd.to_datetime(data['date_of_purchase'], errors='coerce')

print(data)

#>>Output: 5 people in rows with col of name, age, purchase_amount, and date_or_purchase
#   John is not repeated TWICE; ONLY SHOW AS ONCE
#   Can see output in image on site

"""Step 2: Data integration
combining data multiple sources create unified dataset. Often necessary when data collected from different source systems
Techniques incl:
        Schema matching (align fields + data structures diff source => consistency)
        Data deduplication (id and remove duplicate across multiple datasets) """
#Ex: customer data multiple databases -> single view

#Create two manual datasets
data1 = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'name': ['John', 'Jane', 'Jack'],
    'age': [28, 34, 29]

})

data2 = pd.DataFrame({
    'customer_id': [1, 3, 4],
    'purchase_amount': [100.5, 85.3, 45.0],
    'purchase_date': ['2023-12-01', '2023-12-02', '2023-12-03']
})

#Merging datasets on a common key 'customer_id'
merged_data = pd.merge(data1, data2, on='customer_id', how='inner')

print(merged_data)

#>>Output: 5 columns that only have John and Jack once

"""Step 3: Data transformation
Data transfomation converts data into formats suitable for analysis, ML, or mining
Ex:
    Scaling and Normalization (adjust numeric values to common scale necessary algs rely distance metrics)
    Encoding categorical variables (converting categorical data into numerical val using one-hot or label encoding techniques)
    Feature engineering and extraction (Creating new features or selecting important ones to improve model performance)"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder

#Creating a manual dataset
data = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B']
    'numeric_column': [10, 15, 10, 20, 15]
})

#Scaling numeric data
scaler = StandardScaler()
data['scaled_numberic_column'] = scaler.fit_transform(data[['numeric_column']])

#Encoding categorical variables using one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
encoded_data = pd.DataFrame(encoder.fit_transform(data[['category']]),
                            columns=encoder.get_feature_names_out(['category']))

#Concatenating the encoded data with the original dataset
data = pd.concat([data, encoded_data], axis=1)

print(data)

#>>Output: Graph of a 6x7 with 5 values and category, numeric_co, scaled_numeric_col, cat_A, cat_B, and cat_C columns
#table is on site

"""Step 4: Data reduction
Simplifies dataset by reducing number features or records while preserve essential info.
Helps speed up analysis + model train without sacrifice accuracy
Techniques:
    Feature selection: Choose most important features contribute analysis or model performance
    Principal component analysis (PCA): dimensionality reduction tech
        transform data to lower dimensional space
    Sampling methods: Reduce size dataset by selcting representative samples
        useful handle large datasets"""

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

# Creating a manual dataset
data = pd.DataFrame({
    'feature1': [10, 20, 30, 40, 50],
    'feature2': [1, 2, 3, 4, 5],
    'feature3': [100, 200, 300, 400, 500],
    'target': [0, 1, 0, 1, 0]
})

#Feature selection using SelectKBest
selector = SelectKBest(chi2, k=2)
selected_features = selector.fit_transform(data[['feature1', 'feature2', 'feature3']], data['target'])

#Printing selected features
print("Selected features (SelectKBest):")
print(selected_features)

#Dimensionality reduction using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data[['feature1', 'feature2', 'feature3']])

#Printing PCA results
print("PCA reduced data:")
print(pca_data)

#>>Output: (on site), gave two samples for each (stil a little of a black box as fara as how PCA works)