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