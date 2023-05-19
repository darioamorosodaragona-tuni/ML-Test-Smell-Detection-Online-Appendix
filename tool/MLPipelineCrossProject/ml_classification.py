import numpy as np
import pandas as pd
import pickle
import json
import os
from sys import platform

from matplotlib import pyplot as plt
from scipy.stats import uniform
from sklearn.model_selection import LeaveOneGroupOut, TimeSeriesSplit, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, accuracy_score, \
    f1_score, matthews_corrcoef, precision_recall_curve, auc, average_precision_score, PrecisionRecallDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

################################################################
#
#        Hyperparameters Optimization
#
################################################################

optimization_available = ["none", "randomsearch"]


def parameters_space(model):
    if model == 'randomforest':
        n_estimators = np.arange(100, 2000, step=100)
        max_features = ["sqrt", "log2"]
        max_depth = list(np.arange(10, 100, step=10)) + [None]
        min_samples_split = np.arange(2, 10, step=2)
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        param_space = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
        }

    if model == 'adaboost':
        n_estimators = np.arange(100, 1000, step=100)
        learning_rate = [0.001, 0.01, 0.1]

        param_space = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate
        }

    if model == 'multilayerperceptron':
        hidden_layers_size = [(10, 30, 10), (20,)]
        activation = ['tanh', 'relu']
        solver = ['sgd', 'adam']
        alpha = [0.0001, 0.05]
        learning_rate = ['constant', 'adaptive']

        param_space = {
            "hidden_layer_sizes": hidden_layers_size,
            "activation": activation,
            "solver": solver,
            "alpha": alpha,
            "learning_rate": learning_rate
        }

    if model == 'decisiontree':
        criterion = ['gini']
        max_features = ["auto"]
        max_depth = list(np.arange(10, 100, step=10)) + [None]
        min_samples_split = np.arange(2, 10, step=2)
        min_samples_leaf = [1, 2]

        param_space = {
            "criterion": criterion,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }

    if model == 'svm':
        dual = [True, False]
        C = uniform(1, 10)

        param_space = {
            "dual": dual,
            "C": C
        }

    if model == 'knn':
        leaf_size = list(range(1, 50))
        n_neighbors = list(range(1, 30))
        p = [1, 2]

        param_space = {
            "leaf_size": leaf_size,
            "n_neighbors": n_neighbors,
            "p": p
        }

    if model == "naivebayes":
        smoothing = np.logspace(1, 5, num=50)
        param_space = {
            'var_smoothing': smoothing
        }

    if model == "logisticregression":
        C = np.logspace(-4, 4)
        penalty = ['none', 'l2']
        param_space = {
            "C": C,
            "penalty": penalty
        }

    return param_space


def hyperparam_opt(clf, clf_name, opt_method, X, y):
    if opt_method == "randomsearch":
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        splits_indices = cv.split(X, y)

        random_search = RandomizedSearchCV(clf, parameters_space(clf_name), n_iter=10, cv=splits_indices, scoring="f1",
                                           n_jobs=-1, random_state=0, verbose=0)

        search = random_search.fit(X, y)

        best_param = search.best_params_
        best_score = search.best_score_
        return best_param


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# with open(filepath + ".pkl", 'wb') as f:
#   pickle.dump(best_params, f)

def save_best_params(best_params, filepath):
    with open(filepath + ".json", 'w') as f:
        to_dump = {}
        for k in best_params.keys():
            to_dump[k] = best_params[k]
        f.write(json.dumps(to_dump, cls=NpEncoder))


################################################################
#
#        Classifiers
#
################################################################

# classifiers_available = ["logisticregression", "naivebayes", "adaboost", "xgboost", "randomforest", "svm", "decisiontree", "knn", "multilayerPerceptron"] #"dummyopt", "dummypes", "dummyrandom"]

classifiers_available = ["naivebayes", "adaboost", "randomforest", "svm", "decisiontree", "multilayerperceptron", "dummyrandom"]


def get_clf(param):
    if param == "svm":
        return LinearSVC(verbose=0)
    elif param == "decisiontree":
        return DecisionTreeClassifier(random_state=0)
    elif param == "adaboost":
        return AdaBoostClassifier(n_estimators=100, random_state=0)
    elif param == "multilayerperceptron":
        return MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                             random_state=1)
    elif param == "randomforest":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, verbose=0)
    elif param == "knn":
        return KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
    elif param == "naivebayes":
        return GaussianNB()
    elif param == "logisticregression":
        return LogisticRegression()
    elif param == "dummyopt":
        # always predict as non-exploitable
        return DummyClassifier(strategy="constant", constant=0)
    elif param == "dummypes":
        # always predict as exploitable
        return DummyClassifier(strategy="constant", constant=1)
    elif param == "dummyrandom":
        return DummyClassifier(strategy="uniform")


def scorer(clf, clf_name, X, y):
    tn=0
    tp=0
    fp=0
    fn=0
    inspection_rate =0
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    mcc = 0
    auc_pr = 0

    try:
        y_pred = clf.predict(X)
        print(y_pred)
        cm = confusion_matrix(y, y_pred)
        try:
            tn = cm[0,0]
            fp = cm[0,1]
            fn = cm[1,0]
            tp = cm[1,1]
        except IndexError:
            cm = confusion_matrix(y, y_pred)
            tn = cm[0, 0]
            fp = 0
            fn = 0
            tp = 0
            pass
        inspection_rate = ((tp + fp) / (tp + tn + fp + fn))
        precision = (precision_score(y, y_pred))
        recall = (recall_score(y, y_pred))
        accuracy = (accuracy_score(y, y_pred))
        f1 = (f1_score(y, y_pred))
        mcc = (matthews_corrcoef(y, y_pred))
    except ValueError:
        y_pred = 0


        pass

    # calculate roc curve and auc
    if clf_name == "svm":
        # linearsvc is the only one implementing this method
        try:
            probs = clf.decision_function(X)
            precisionNew, recallNew, thresholds = precision_recall_curve(y, probs)
            auc_pr = auc(recallNew, precisionNew)
        except ValueError:
            probs = 0
            precisionNew=0
            recallNew = 0
            thresholds =0
            auc_pr = 0
            pass
    else:
        try:

            probs = clf.predict_proba(X)[:, 1]
            precisionNew, recallNew, thresholds = precision_recall_curve(y, probs)
            auc_pr = auc(recallNew, precisionNew)
        except ValueError:
            probs = 0
            precisionNew = 0
            recallNew = 0
            thresholds = 0
            auc_pr = 0
            pass

    res = { "tp": [tp], "fp": [fp], "tn": [tn], "fn": [fn], "precision": [precision], "recall": [recall],
            "accuracy": [accuracy], "inspection_rate": [inspection_rate], "f1_score": [f1], "mcc": [mcc], "auc_pr": [auc_pr]}

    return precisionNew, recallNew, pd.DataFrame(res), y_pred


''' if model == 'xgboost':
        n_estimators = np.arange(100, 1000, step=100)

        param_space = {
            "n_estimators": n_estimators
        }'''

'''    elif param == "xgboost":
        return XGBClassifier(n_estimators=100, n_jobs=-1, randomstate=0, use_label_encoder=False)'''
