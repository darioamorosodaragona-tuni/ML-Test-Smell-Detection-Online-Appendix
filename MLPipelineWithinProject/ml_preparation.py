import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer




################################################################
#
#        Data Cleaning
#
################################################################

'''def data_scaling(dataset):
    # remove infinite values and NaN values 
    scaler = MinMaxScaler()
  #  dataset = scaler.fit_transform(dataset)
    dataset = pd.DataFrame(scaler.fit_transform(dataset), index=dataset.index, columns=dataset.columns)

    #dataset = dataset.replace([np.inf, -np.inf], np.nan).fillna(0)
    return dataset

def column_drop(dataset):
    # remove columns which always contain the same value
    dataset = dataset.drop(dataset.std()[(dataset.std() == 0)].index, axis=1)
    return dataset


def vexctorizeToken(token):
    vocabulary_vectorizer = CountVectorizer()
    bow_train = vocabulary_vectorizer.fit_transform(token)
    matrix_token = pd.DataFrame(bow_train.toarray(), columns=vocabulary_vectorizer.get_feature_names())

    return matrix_token'''


################################################################
#
#        Feature Selection
#
################################################################

feature_selection_available = ["vif"]

def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


def vif(X):
    vif1 = calc_vif(X)
    #ig1 = calc_IG(X)
    a=vif1.VIF.max()
    while a > 5:
        maximum_a = vif1.loc[vif1['VIF'] == vif1['VIF'].max()]
        vif1 = vif1.loc[vif1['variables'] != maximum_a.iloc[0,0]]
        vif1 = calc_vif(X[vif1.variables.tolist()])
        a = vif1.VIF.max()
    return vif1.variables.tolist()


def feature_selection(param, X):
    if param == "vif":
        return vif(X)
    


################################################################
#
#        Data Balancing
#
################################################################

data_balancing_available = ["nearmissunder1", "nearmissunder2", "nearmissunder3",
                            "smoteover", "randomover", "randomunder", "adasyn", "borderlinesmote"]

def data_balancing(param, X, y):
    if param == "smoteover":
        #oversample = SMOTE(random_state = 42, k_neighbors = 2)
        oversample = SMOTE(random_state = 42, k_neighbors= 1)
        X, y = oversample.fit_resample(X, y)
    elif param == "adasyn":
        #oversample = ADASYN(n_neighbors=2)
        oversample = ADASYN(n_neighbors=1)
        X, y = oversample.fit_resample(X, y)
    elif param == "borderlinesmote":
        #oversample = BorderlineSMOTE(kind='borderline-1', k_neighbors=2)
        oversample = BorderlineSMOTE(kind='borderline-1', k_neighbors=1)
        X, y = oversample.fit_resample(X, y)
    elif param == "randomover":
        oversample = RandomOverSampler(sampling_strategy='minority')
        X, y = oversample.fit_resample(X, y)
    elif param == "nearmissunder1":
        undersample = NearMiss(version=1, n_neighbors=6)
        X, y = undersample.fit_resample(X, y)
    elif param == "nearmissunder2":
        undersample = NearMiss(version=2, n_neighbors=6)
        X, y = undersample.fit_resample(X, y)
    elif param == "nearmissunder3":
        undersample = NearMiss(version=3, n_neighbors=6)
        X, y = undersample.fit_resample(X, y)
    elif param == "randomunder":
        undersample = RandomUnderSampler(sampling_strategy='majority', random_state=42)
        X, y = undersample.fit_resample(X, y)
    return X, y