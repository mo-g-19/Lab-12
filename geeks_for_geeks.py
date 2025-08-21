#Basically notes from https://www.geeksforgeeks.org/machine-learning/data-preprocessing-machine-learning-python/
#Goes through different steps; formatted more as notes than working code
#   Because that's what it is first, notes

"""Intro: data reprocessing important step data science transform raw data
into clean structured format for analysis. Involves handling missing values,
normalizing data, and encoding variables. Master => reliable insight accurate predict + effective decision-making
    pre-processing -> transformation applied data before feed algorithm
"""

"""Steps in Data Preprocessing
Step 1: Import necessary libraries"""
#importing libraries
import pandas as pd
import scipy
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns              #I do not have that library downloaded on this computer
import matplotlib.pyplot as plt    #I have not downloaded/figured out how to connect matplot to this virtual enviroment; don't want to have to redownload matplot

"""
Step 2: Load dataset - They give a specific dataset in a link (https://media.geeksforgeeks.org/wp-content/uploads/20250115110111213229/diabetes.csv)"""
#load dataset (it's the one line csv dataset load)
df = pd.read_csv('Geeksforgeeks/Data/diabetes.csv')
print(df.head())
#>>Output => image see on the website (columns of 5 different people)

"""
    Check the data info"""
df.info()
#>>Output => See the column number, associated name, the count, and Dtype (all non-null)
#Check null val using
df.isnull().sum()
#>>Output => column name and total number of null values for said value (image on site)

"""
Step 3: Statistical Analysis"""
#give descriptive overview of dataset
df.describe()
#   table show count, mean, standard deviation, min, 25%, 50%, 75%, and max val each col
#       find Insulin, Pregnancies, BMI, BloodPressure col => outliers

"""
    Check outliers"""
#Box Plots
#In: 9 rows, 1 column
#   dpi related to image size https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
#       Dots per inches: refers how many pixels figure comprises; default in matplotlib is 100
#       dpi scales elements (larger dpi => magnify glass)
#   figsize also realted to image size (use same Stack Overflow page for explain)
#       determines size of figure in inches, gives amount space axes (and other elements) hae inside fig
#           default: (6.4, 4.8); larger => longer texts, more axes or more ticklabels shown
#Return: fig (figure), axs (array of Axes - an array of Axes objects (more one subplot) or just one)
#       dimension of result array controlled squeeze keyword
fig, axs = plt.subplots(9, 1, dpi=95, figsize=(7,17))

"""
    Drop outliers"""
#*Insuline*
#Identify the quartiles
q1, q3 = np.percentile(df['Insuline'], [25, 75])
#Calculate the interquartile range
iqr = q3 - q1
#Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
#Drop the outliers
clean_data = df[(df['Insuline'] >= lower_bound)
                & (df['Insuline'] <= upper_bound)]


#*Pregnencies*
#Identify the quartiles
q1, q3 = np.percentile(df['Pregnencies'], [25, 75])
#Calculate the interquartile range
iqr = q3 - q1
#Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
#Drop the outliers
clean_data = df[(df['Pregnencies'] >= lower_bound)
                & (df['Pregnencies'] <= upper_bound)]


#*Age*
#Identify the quartiles
q1, q3 = np.percentile(df['Age'], [25, 75])
#Calculate the interquartile range
iqr = q3 - q1
#Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
#Drop the outliers
clean_data = df[(df['Age'] >= lower_bound)
                & (df['Age'] <= upper_bound)]


#*Glucose*
#Identify the quartiles
q1, q3 = np.percentile(df['Glucose'], [25, 75])
#Calculate the interquartile range
iqr = q3 - q1
#Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
#Drop the outliers
clean_data = df[(df['Glucose'] >= lower_bound)
                & (df['Glucose'] <= upper_bound)]


#*BloodPressure*
#Identify the quartiles
q1, q3 = np.percentile(df['BloodPressure'], [25, 75])
#Calculate the interquartile range
iqr = q3 - q1
#Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
#Drop the outliers
clean_data = df[(df['BloodPressure'] >= lower_bound)
                & (df['BloodPressure'] <= upper_bound)]

#*BMI*
#Identify the quartiles
q1, q3 = np.percentile(df['BMI'], [25, 75])
#Calculate the interquartile range
iqr = q3 - q1
#Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
#Drop the outliers
clean_data = df[(df['BMI'] >= lower_bound)
                & (df['BMI'] <= upper_bound)]


#*DiabetesPedigreeFunction*
#Identify the quartiles
q1, q3 = np.percentile(df['BMIDiabetesPedigreeFunction'], [25, 75])
#Calculate the interquartile range
iqr = q3 - q1
#Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
#Drop the outliers
clean_data = df[(df['BMIDiabetesPedigreeFunction'] >= lower_bound)
                & (df['BMIDiabetesPedigreeFunction'] <= upper_bound)]

"""
    Correlation"""
#Correlatoin
corr = df.corr()

#Matplot + Seaborn visual
plt.figure(dpi=130)
sns.heatmap(df.corr(), annot=True, fmt= '.2f')
plt.show()
#   Can compare by single columns in descending order
corr['Outcome'].sore_alues(ascending = False)
#>>Output: score of how strongly a column correlates to 'Outcome' (of course a 1.000 for Outcome col)

"""
    Check Outcomes Proportionally"""
#Pie chart of people with vs without diabetes
plt.pie(df.Outcome.value_counts(), 
        labels = ['Diabetes', 'Not Diabetes'],
        autopct = '%.f', shadow=True)
plt.title('Outcome Proportionality')
plt.show()

"""
Separate independent features + target values"""
#seperate array into input and output componenets
X = df.drop(columns = ['Outcome'])
Y = df.Outcome

"""
Step 4: Normalization or Standardization
    Normalization
        well when features have different scales + algorithm being used sensitive scale of features (kNN (nearest neighbor) or neural networks)
        rescale data using MniMaxScalar
            scales data so each feature range [0, 1]"""
# initializing the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#Learning the statistical parameters for each of data and transforming
rescaledX = scaler.fit_transform(X)
rescaledX[:5]
#>>Output: a 2D array of 8 col (match 8 categories) and 5 rows (instructed by [:5])
#           0 <= all values < 1

"""
    Standardization
        well when features have normal distribution or alg not sensitive scale of features
        useful to transform attributes with Gaussian distribution and differing means + standad deviations to standard Gaussian distribution with mean of 0 and standard deviation of 1
        can standardized data using StandardScaler class """
from sklearn.preprocessing import StandardScaler

scaler = StadardScaler().fix(X)
rescaledX = scaler.transform(X)
escaledX[:5]
#>>Output: a 2D array of 8 col (Match 8 categories) and 5 rows (from [:5])
#           wide range of values (from negative to positive); relatively small numbers just because of data relating

