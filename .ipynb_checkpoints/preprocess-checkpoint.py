"""This module contains the preprocessing functions required to build a preprocessing pipeline.
These functions are specific to the advanced house price predicting problem"""

import eda
import pandas as pd
import numpy as np


# A function to impute the missing values of the categorical features
def impute_cat_features(data, fill_with='missing'):
    """Fills the Null values of the categorical data with a string specified by fill_with
    Args:
        data: DataFrame, the dataset.
        fill_with: String, a word or character to replace the NaN values.
    """
    df = data.copy()
    # Specify a list of categorical features that have NaNs.
    features = eda.feature_with_NaN(eda.categorical_data(df))
    # Replacing the NaNs with the fill_with
    for feature in features:
        df[feature].fillna(fill_with,inplace=True)
    return df

# A function to impute the missing values of the numerical features
def impute_numeric_data(data, method='median'):
    """Fill the missing values of the numerical features of the data. In adition it
    creates new binary features for each numeric feature which has missing values, replacing
    the NaNs with 1, otherwise with 0.
    Args:
        data: A DataFrame
        method: one of the ['median', 'mean', 'mode'], indicating how the imputation should
        occur. In case of existing two or more modes the first one is selected.
    """    
    df = data.copy()    
    # Numerical features which have missing values
    features = eda.feature_with_NaN(eda.numeric_data(df))
    for feature in features:
        # Getting the column index of the feature
        idx = df.columns.get_loc(feature)
        # Creating a new binary feature beside the original feature column, replacing the 
        # NaN values with 1, and the non-NaNs with 0
        binary_values = np.where(df[feature].isnull(), 1, 0)
        new_col_name = feature+'_binary_NaN'
        # Inserting the new column at the specific index,exactly beside the original column
        df.insert(loc=idx+1, column=new_col_name, value=binary_values)
        # Filling the missing values in the original features
        if method=='median':
            df[feature].fillna(value=df[feature].median(), inplace=True)
        elif method=='mean':
            df[feature].fillna(value=df[feature].mean(), inplace=True)
        elif method=='mode':
            df[feature].fillna(value=df[feature].value_counts().index[0], inplace=True)
    return df 

# A function to create new features from the historic ones.
def create_age_feature(data, sold_year_col='YrSold'):
    """Transform the historic features into an age feature, by subtracting the feature
    from the sold_year feature.
    Args:
        data: DataFrame, the dataset
        sold_year_col: String, name of the column indicating the year in which
        house was sold.
    """
    df = data.copy()
    # Getting the list of historic features
    features = eda.historic_data(df).columns
    # Replacing each feature by the age values.
    for feature in features:
        if feature != sold_year_col:
            df[feature] = df[sold_year_col] - df[feature]
    return df   

# Transform the continous features into logarithmic scale
def log_normalize(data):
    """Transforms the continous features into logarithmic scale.
    This transformation is applied ony on the features which have positive values.
    """
    df = data.copy()
    # Getting the list of the continous features.
    features = eda.numeric_continous(df).columns
    # Replacing the feature with its log-transformed version.
    for feature in features:
        df[feature] = eda.log_transform(df[feature])        
    return df 

# Identifying the rare categories within ach feature.
def rare_categories(data, threshold=0.01, replace_with='rare_value'):
    """Replaces the rare classes of the categorical features with replace_with.
    It computes the percentage of each class of a feature. If it is less than the
    threshold, then it will be replaced with replace_with.
    Args:
        data: DataFrame, the dataset
        threshold: float, indicating a fraction threshold to select the 
                    rare categories of a feature.
        replace_with: string, a keyword to replace the rare categories.
    """
    df = data.copy()
    # Getting the list of categorical features.
    features = eda.categorical_data(df).columns
    # Iterating over the features list
    for feature in features:
        # Creating a dataframe. Its index is the unique categories of the feature 
        # and its column's value indicates the fraction of categories.
        df_temp = pd.DataFrame(df[feature].value_counts()/len(df))
        # Creating a mask to select the rare categories based on the threshold.
        mask = df_temp[feature] <= threshold
        # Specifying the list of rare categories
        rare_classes = df_temp[mask].index
        # Replacing the rare categories with the string specified by replace_with        
        df[feature]= df[feature].replace(rare_classes, replace_with)
    return df

# A function to convert the categorical features into numeric.
def cat_to_numeric(data):
    """Maps the categorical features into numeric.
    It first computes the counts of each category of a feature.
    Then it assigns an integer to the sorted categories. The most frequent one gets 1
    and the least frequent one receives the length of the unique values of the feature
    """
    df = data.copy()
    # Getting the list of categorical features
    features = eda.categorical_data(df).columns
    # Iterating over the list of features.
    for feature in features:
        # Sorting the categories based on their counts and store the sorted
        # categories in a list.
        categories = df[feature].value_counts().index
        # Specify a dictionry to map the ategories into integers.
        map_dict = {k:v+1 for v,k in enumerate(categories)}
        # Performing the map operation
        df[feature] = df[feature].map(map_dict)
    return df

# Dropping unuseful feature.
def drop_feature(data, to_drop=['Id']):
    df = data.copy()
    return df.drop(to_drop, axis=1)