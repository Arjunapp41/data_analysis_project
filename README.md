%pip install scikit-learn==1.4.1.post1 seaborn==0.13.2 fasteda==1.0.1
import random
random.seed(2024)

import missingno as msno
import numpy as np
from scipy.stats import shapiro
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_style('white')
diabetes_X, diabetes_y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
diabetes = pd.concat([diabetes_X, pd.Series(diabetes_y)], axis=1).rename({0: 'target'},axis=1)
diabetes.sample(5)
# Initialize the one-hot encoder
enc1 = OneHotEncoder(handle_unknown='ignore', drop=None)

# One-hot encode 'sex'; the output is a numpy array
encoded_sex = enc1.fit_transform(diabetes[['sex']]).toarray()

# Convert numpy array to pandas DataFrame with columns names based on original category labels
encoded_sex = pd.DataFrame(encoded_sex, columns=['sex' + str(int(x)) for x in enc1.categories_[0]])

# Horizontally concatenate the original 'diabetes' data set with the two one-hot columns
diabetes = pd.concat([diabetes, encoded_sex], axis=1)

# Sample 10 rows. Print only the 'sex', 'sex1', and 'sex2' columns for simplicity
diabetes[['sex', 'sex1', 'sex2']].sample(10)
# Drop 'sex' and 'sex2'
diabetes = diabetes.drop(['sex', 'sex2'], axis=1)
# Reorder columns to have 'sex1' where 'sex' used to be
diabetes = diabetes.loc[:, ['age', 'sex1', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'target']]
# Print a sample of 5 rows
diabetes.sample(5)
# Make a train-test split
X_train, X_test, y_train, y_test = train_test_split(diabetes.iloc[:, :-1], diabetes.iloc[:, [-1]], test_size=0.33, random_state=2024)
diabetes.head()
diabetes.tail()
diabetes.describe()
# linear regression dropping NANs

# Get NAN indices
nonnan_train_indices = X_train.index[~X_train.isna().max(axis=1)]
nonnan_test_indices = X_test.index[~X_test.isna().max(axis=1)]

# Fit an instance of `LinearRegression`
reg = LinearRegression().fit(X_train.loc[nonnan_train_indices], y_train.loc[nonnan_train_indices])
# Generate predictions
pred = reg.predict(X_test.loc[nonnan_test_indices])
# Calculate the root mean squared error
root_mean_squared_error(y_test.loc[nonnan_test_indices],pred)

diabetes_default = load_diabetes()
 # linear regression with mean fill

# Get NAN indices
nonnan_train_indices = X_train.index[~X_train.isna().max(axis=1)]
nonnan_test_indices = X_test.index[~X_test.isna().max(axis=1)]

# Initialize the simple imputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# Fit the simple imputer using the training data
imp_mean.fit(X_train)
# Actually mean fill the training data
X_train_mean_filled = imp_mean.transform(X_train)

# Fit an instance of `LinearRegression`
reg = LinearRegression().fit(X_train_mean_filled, y_train)
# Generate predictions
pred = reg.predict(X_test.loc[nonnan_test_indices])
# Calculate the root mean squared error
root_mean_squared_error(y_test.loc[nonnan_test_indices],pred)
# linear regression with median fill

# Get NAN indices
nonnan_train_indices = X_train.index[~X_train.isna().max(axis=1)]
nonnan_test_indices = X_test.index[~X_test.isna().max(axis=1)]

# Initialize the simple imputer
imp_median = ### REPLACE WITH YOUR CODE ###
# Fit the simple imputer using the training data
imp_median.fit(X_train)
# Actually median fill the training data
X_train_median_filled = imp_median.transform(X_train)

# Fit an instance of `LinearRegression`
reg = LinearRegression().fit(X_train_median_filled, y_train)
# Generate predictions
pred = reg.predict(X_test.loc[nonnan_test_indices])
# Calculate the root mean squared error
root_mean_squared_error(y_test.loc[nonnan_test_indices],pred)
