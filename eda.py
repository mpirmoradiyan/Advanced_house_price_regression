"""This module is specific to House-price predictibg problem.
It includes functions for preprocessing, EDA, and feature engineering"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Functions to deal with the missing values
def feature_with_NaN(data):
    """A function which returns the list of columns which have Null values"""
    features = [col for col in data.columns if data[col].isna().sum() !=0]
    return features

def has_NaN(data):
    """Prints the number of NaN values for every feature which has NaN values."""
    for col in feature_with_NaN(data):
        print(f"{col}: {data[col].isna().sum()} NaN valus, {(100*data[col].isna().sum())/len(data):.2f}% of the data samples")


def make_binary(data, target_var='SalePrice'):
    """This fumction returns a dataframe consisting of features which have NaN values
    and the target variable. The features are transformed to binary ones 
    based upon whether their values are NaN or not.
    ##############
    Args:
        data: A dataframe which is the dataset
        target_var: string, target variable in the dataset
    ##############
    return:
        A dataframe, including binary features, 1 for the NaN values and else for the non NaNs,
        and the original target variable.
    """
    features = feature_with_NaN(data)
    df = data.loc[:,features]
    df = pd.DataFrame(np.where(df.isna(),1,0), columns=features)
    df[target_var]=data[target_var]
    return df

##################################################################################################

# Functions to distinguish different feature types
def numeric_data(data):
    """Returns a dataframe filtered out the non-numeric features from the input data."""
    # Dropping ID column as it is non informative feature
    df = data.drop('Id',axis=1)  
    return df.select_dtypes(include='number')

def numeric_discrete(data, target_var="SalePrice", threshold=25, include_target_var=True):
    """Returns a datframe consisting of the discrete numerical features.
    Args:
        data: Dataframe
        threshold: int, determins the threshold of the number of unique values
        of a feature for deciding if the feature is discrete or not.
    """
    # Dropping the historic features.
    df = numeric_data(data).drop(historic_data(data).columns,axis=1) 
    # Check if the feature is discrete considering the threshold.
    disc_features =[col for col in df.columns if df[col].nunique()<threshold]
    # Filtering the data to contain only the discrete features.
    df=df.filter(items=disc_features)
    # Check if it is required to include the target variable.
    if include_target_var:
        df[target_var]=data[target_var]
    return df

def numeric_continous(data, target_var="SalePrice", threshold=25, include_target_var=True):
    """Returns a datframe consisting of the discrete numerical features.
    Args:
        data: Dataframe
        threshold: int, determins the threshold of the number of unique values
        of a feature for deciding if the feature is discrete or not.
    """
    # Dropping the historic features.
    df = numeric_data(data).drop(historic_data(data).columns,axis=1) 
    # Check if the feature is continous considering the threshold.
    cont_features =[col for col in df.columns if df[col].nunique()>=threshold]
    # Filtering the data to contain only the continous features.
    df=df.filter(items=cont_features) 
    if not include_target_var:
        df.drop(target_var,axis=1, inplace=True)       
    return df   
    


def historic_data(data,target_var='SalePrice', include_target_var=False):
    """This function filters the data by keeping those features which shows a history.
    The features whose names contains 'Year' or 'Yr' will be kept.
    The target variable also will be added.
    """
    # Filtering the data to contain only the those features which take on years as their values.
    df = data.filter(regex=f'Yr|Year')
    # Check if it is required to include the target variable.
    if include_target_var:
        df[target_var]=data[target_var]
    return df


def categorical_data(data, target_var='SalePrice',include_target_var=False):
    """Returns a dataframe containing the non-numeric features."""
    df = data.select_dtypes(exclude='number')
    if include_target_var:
        df[target_var]=data[target_var]
    return df

def cat_features_uniques(data):
    """Prints the categorical features along with their unique values."""
    df = categorical_data(data,include_target_var=False)
    for feature in df.columns:
        print(f"Feature name: {feature}")
        print(f"\t Number of unique values:{df[feature].nunique()}")
        print(f"\t Unique values: {df[feature].unique()}\n")

############################################################################################
# Visualize functions
def plot_binary_df(data, target_var='SalePrice',figsize=(7,60)):
    """This function returns barplots of the binary features output by make_binary
    function. It also makes a hist plot of the target variable hued for every feature.
    
    """
    # list of features
    features = data.drop(target_var,axis=1).columns
    # length of the features to determine number of rows and columns for the figure
    n = len(features)
    ncol = 2   #number of columns of the figure
    #nrow = int(np.ceil(n/ncol))  #number of rows of the figure
    
    #Specify the figure and subplots configuration
    fig,axs = plt.subplots(n,ncol,figsize=figsize)
    
    for i,feature in zip(list(range(n)),features):
        sns.barplot(data=data.groupby(feature)[target_var].median().reset_index(),      
                    x=feature, y=target_var, ax=axs[i,0])
#         sns.barplot(data=data, x=feature, y=target_var, ax=axs[i,0])
        axs[i,0].set_ylabel('SalePrice(Median)')
        sns.histplot(data=data, x=target_var, hue=feature,
                     element="poly",ax=axs[i,1])
    # Deleting the empty axes of the subplots
#     for ax in axs.flat:
#         if not ax.lines:
#             #ax.set_visible(False)
#             plt.delaxes(ax)
    fig.tight_layout();


def age_price_plot(data):
    """This function calculate the age of the building considering the 
    built year and the sold year and make a scatter plot 
    for every historic_feature-SalePrice pair.
    """
    # The features which reflects a history.
    historic_features = historic_data(data).columns
    # Specifying appropriate names for the x-axis
    age_names = ['House_age', 'Remod_age','Garage_age']
    # Defining the figure configurations
    fig,axs = plt.subplots(3,1,figsize=(10,12))    
    for ax,col,i in zip(axs,historic_features, list(range(len(age_names)))):
        if col != 'YrSold':
            x= data['YrSold']- data[col]
            sns.scatterplot(x=x, y=data['SalePrice'],ax=ax)
            ax.set_xlabel(age_names[i])
            

def discrete_price_barplot(data, target_var='SalePrice'):
    """Plots the barplots. The y-axis is the median of the SalePrice for each class of features."""
    df = numeric_discrete(data)
    features = df.drop(target_var,axis=1).columns
    fig,axs = plt.subplots(len(features),1,figsize=(9,65))
    for ax,feature in zip(axs.flat,features):        
        sns.barplot(data=df.groupby(feature)[target_var].median().reset_index(),
                    x=feature,
                    y=target_var,
                    ax=ax)
        ax.set_ylabel('SalePrice(Median)')
    fig.tight_layout()


def continous_histplot(data, target_var='SalePrice'):
    """Returns the histograms of the continous features"""
    df = numeric_continous(data, include_target_var=True)
    features = df.columns
    ncol=2
    nrow=int(np.ceil(len(features)/ncol))
    fig,axs = plt.subplots(nrow,ncol,figsize=(14,28))
    for ax,feature in zip(axs.flat,features):        
        sns.histplot(data=df, x=feature,ax=ax)
    fig.tight_layout() 

def continous_saleprice_plot(data, target_var='SalePrice'):
    """Returns the joinplot of the continous features vs. the target variable"""
    # Specifying a dataframe of continous variables.
    df = numeric_continous(data, include_target_var=True)
    # Specifying a list of continous features.
    features = df.drop(target_var, axis=1).columns   
    for feature in features:        
        sns.jointplot(data=df, x=feature,y=target_var, marker='o',s=50,
                     marginal_kws={'bins':50, 'fill':False},
                     marginal_ticks=True
                     )
###########################################################################################
def log_transform(feature):
    """Applies a logarithmic transformation to the numeric feature if it does not have zero values."""
    if (feature <=0).any():
        pass
        #feature=feature.map({0:0.0000000001})
    else:
        feature=feature.map(lambda x:np.log(x))
    return feature