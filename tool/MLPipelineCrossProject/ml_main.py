from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from ml_utils import get_params, id_string, out_dir
from ml_preparation import feature_selection, data_balancing
from ml_classification import get_clf, hyperparam_opt, scorer, save_best_params
from datetime import datetime
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
import pickle
from halo import Halo
import json
import gc
import os
import numpy as np

print("________________________________")
print("Starting")
# get pipeline customization parameters and setup output directories
params = get_params()
k = params["k"]
# k = 10
start_time = datetime.now()
time_str = start_time.strftime("%Y%m%d_%H%M%S")
job_id = id_string(params)
out_dir = out_dir(job_id + "__" + time_str)

precision_list = []
recall_list = []
# fpr_list = []
# tpr_list = []

print("Started: " + start_time.strftime("%d %m %Y H%H:%M:%S"))
print("Results will be saved to dir: " + out_dir)
# get dataset
dataset_dir = "/yourpath/ML-Test-Smell-Detection-Online-Appendix/dataset/entireDatasets/" + params[
    "data"] + ".csv"  # Insert here your path

# df_columns = ["idProject","nameProject","productionClass","testCase","NMC","SimilaritiesCoefficient","probabilityEagerTest",
#           "isEagerTest","isEagerTestManual"]

# this line is for RQ3
df = pd.read_csv(dataset_dir)

num_unique = df['nameProject'].nunique()
print(num_unique)

print("Loaded dataset of size: ")
print(df.shape)
print("Splitting dataset...")

project_names = df["nameProject"].unique()

print(type(project_names))

print(project_names)


# split dataset in k folds
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

#X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].copy()  # this line is for EagerTestPrediction
X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, ]].copy()

X = df.iloc[:, df.columns != 'isMysteryGuestManual'].copy()
y = df['isMysteryGuestManual']

#folds = kfold.split(X, y)

i = 0
cols = df.select_dtypes([np.int64, np.int32]).columns
df[cols] = np.array(df[cols], dtype=float)
# df[cols] = df[cols].astype(float)

print_header = True

for project_name in project_names:
    perf_df = pd.DataFrame([])

    for train_index, test_index in kfold.split(X, y):
    # Get the training and testing data for this fold
        X_train_project, X_test_project = X.iloc[train_index], X.iloc[test_index]
        y_train_project, y_test_project = y.iloc[train_index], y.iloc[test_index]

    #for project_name in project_names:
        # Split the data into training and testing sets for this project
        X_train = X_train_project[X_train_project["nameProject"] != project_name]
        X_test = X_test_project[X_test_project["nameProject"] == project_name]
        y_train = y_train_project[X_train_project["nameProject"] != project_name]
        y_test = y_test_project[X_test_project["nameProject"] == project_name]

        print(y_test)

        #y_train_project = y_train_project["isEagerTestManual"]
        y_train = y_train.astype(int)

        X_train = X_train.drop(
            columns=["idProject", "nameProject", "productionClass", "testCase", "isMysteryGuest"])

        testset_sample = X_test[
            ["idProject", "nameProject", "productionClass", "testCase", "isMysteryGuest"]]

        #y_test_project = y_test_project["isEagerTestManual"]
        y_test = y_test.astype(int)

        X_test = X_test.drop(
            columns=["idProject", "nameProject", "productionClass", "testCase", "isMysteryGuest"])


        print("Data cleaning")

        columns_to_retain = X_train.columns
        X_test = X_test[columns_to_retain]
        if not params["feature_sel"] == "none":
            print("Round " + str(i + 1) + " of " + str(k) + ": " + "Feature selection")
            columns_to_retain = feature_selection(params["feature_sel"], X_train_project)
            X_train = X_train[columns_to_retain]
            X_test = X_test[columns_to_retain]

        data = []
        data = [columns_to_retain, mutual_info_classif(X_train[columns_to_retain], y_train, discrete_features=True)]

        data_T = pd.DataFrame(data).T
        data_T.columns = ["variable", "value"]

        data_filter = data_T[data_T.value > 0]
        X_train = X_train[data_filter.variable]
        X_test = X_test[data_filter.variable]

        with open(out_dir + "IG/" + str(i) +"_" +project_name+ ".txt", 'w') as f:
            dfAsString = data_T.to_string(header=False, index=False)
            f.write(dfAsString)
        del columns_to_retain
        gc.collect()

        # fix bug with numpy arrays
        X_train = X_train.values
        y_train = y_train.values.ravel()
        X_test = X_test.values
        y_test = y_test.values.ravel()

        # data balancing

        if not params["balancing"] == "none":
            print("Data balancing")
            X_train, y_train = data_balancing(params["balancing"], X_train, y_train)

        # classifier
        if not params["classifier"] == "none":
            clf_name = params["classifier"]
        else:
            clf_name = "dummy_random"
        clf = get_clf(clf_name)

        # hyperparameter opt

        if not params["optimization"] == "none" and not clf_name.startswith("dummy"):
            print("Round " + str(i + 1) + " of " + str(k) + ": " + "Hyperparameters optimization")
            best_params = hyperparam_opt(clf, clf_name, params["optimization"], X_train, y_train)
            save_best_params(best_params, out_dir + "best_params/" + str(i))
            clf.set_params(**best_params)

        # validation
        print("Training")
        clf.fit(X_train, y_train)

        del X_train
        del y_train
        gc.collect()

        print("Testing")
        # fpr, tpr, res, y_pred = scorer(clf, clf_name, X_test, y_test)
        precisionNew, recallNew, res, y_pred = scorer(clf, clf_name, X_test, y_test)
        y_pred = pd.DataFrame(y_pred, columns=["prediction"], index=testset_sample.index)
        y_pred.replace({0.0: False, 1.0: True}, inplace=True)

        agreement = testset_sample.join(y_pred)
        mode = 'w' if print_header else 'a'
        agreement.to_csv(out_dir + "resultTestCase/"+"resultForTestCase_"+project_name+".csv", mode=mode, header=print_header)
        print_header = False
        pyplot.figure()
        pyplot.plot(recallNew, precisionNew)
        pyplot.savefig(out_dir + "roc_curves/" + str(i) + ".png")
        precision_list.append(precisionNew)
        recall_list.append(recallNew)
        # fpr_list.append(fpr)
        # tpr_list.append(tpr)
        perf_df = pd.concat([perf_df, res], ignore_index=True)


        del X_test
        del y_test

        gc.collect()


    print("Saving performance")

#project = perf_df["nameProject"]
    sumTN = perf_df['tn'].sum()
    sumFP = perf_df['fp'].sum()
    sumFN = perf_df['fn'].sum()
    sumTP = perf_df['tp'].sum()
    meanPR = perf_df['precision'].mean()
    meanRC = perf_df['recall'].mean()
    meanACC = perf_df['accuracy'].mean()
    meanIR = perf_df['inspection_rate'].mean()
    meanF1 = perf_df['f1_score'].mean()
    meanMCC = perf_df['mcc'].mean()
    meanAUC_PR = perf_df['auc_pr'].mean()
    # meanAUC = perf_df['auc_roc'].mean()

    list = [sumTP, sumFP, sumTN, sumFN, meanPR, meanRC, meanACC, meanIR, meanF1, meanMCC, meanAUC_PR]  # , meanAUC]
     #perf_df  = pd.read_csv('performance.csv')
    perf_df = perf_df.append(pd.Series(list, index=perf_df.columns[:len(list)]), ignore_index=True)

#perf_df = pd.concat([pd.Series(project_names, name = "Project"),perf_df], axis = 1)
    perf_df.to_csv(out_dir + "resultPerformance/" + "performance_"+project_name+".csv")

    del perf_df

'''pyplot.figure()
for i in range(8):
    pyplot.plot(recallNew[i], precisionNew[i])
pyplot.savefig(out_dir + "pr_curves/all.png")'''

end_time = datetime.now()
print(params)

print("Ended: " + end_time.strftime("%d %m %Y H%H:%M:%S"))
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time}")
with open(out_dir + "elapsed_time.json", 'w') as f:
    f.write(json.dumps({"elapsed_time": f"{elapsed_time}"}))

print("________________________________")

