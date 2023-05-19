import csv

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, f1_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from ml_utils import get_params, id_string, out_dir
from ml_preparation import feature_selection, data_balancing
from ml_classification import get_clf, hyperparam_opt, scorer, save_best_params
import gc
from matplotlib import pyplot



# Load the dataset
data = pd.read_csv("/Users/valeriapontillo/Documents/ML-Test-Smell-Detection-Online-Appendix/eagerTestTotal.csv")

#dataset_dir = "/Users/valeriapontillo/Documents/ML-Test-Smell-Detection-Online-Appendix/" + params[
 #   "data"] + ".csv"  # Insert here your path

#this line is for RQ3
#data = pd.read_csv(dataset_dir)

# Encode the "nameProject" column as integer labels
cols_to_drop = ["idProject", "productionClass", "testCase", "isEagerTest"]
data = data.drop(cols_to_drop, axis=1)

le = LabelEncoder()
data["nameProject"] = le.fit_transform(data["nameProject"])

project = [(label, value) for label, value in zip(le.classes_, le.transform(le.classes_))]

dict_1 = {item[1]: item[0] for item in project}


# Define the features and target variable
#X = data.drop("isEagerTestManual", axis=1)
X = data.drop("isEagerTestManual", axis=1)
y = data["isEagerTestManual"]

#cols = data.select_dtypes([np.int64, np.int32]).columns
#data[cols] = np.array(data[cols], dtype=float)

# Define the number of folds for the cross-validation
n_folds = 10

# Define the cross-validation strategy
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize a list to store the accuracy scores
scores_accuracy = []
scores_precision = []
scores_f1 = []
project_test = []

precision_list = []
recall_list = []

# Get the unique project names in the testing data
project_names = data["nameProject"].unique()

cols = data.select_dtypes([np.int64, np.int32]).columns
data[cols] = np.array(data[cols], dtype=float)
#df[cols] = df[cols].astype(float)

print_header = True

for train_index, test_index in kf.split(X, y):
    # Get the training and testing data for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf = DecisionTreeClassifier(random_state=42)


for project_name in project_names:
    # Split the data into training and testing sets for this project
    X_train_project = X_train[X_train["nameProject"] != project_name]
    X_test_project = X_test[X_test["nameProject"] == project_name]
    y_train_project = y_train[X_train["nameProject"] != project_name]
    y_test_project = y_test[X_test["nameProject"] == project_name]

    clf.fit(X_train_project, y_train_project)

    # Make predictions on the testing data for this project
    try:
        y_pred_project = clf.predict(X_test_project)
    # Calculate the accuracy score for this project and add it to the scores list
        accuracy = accuracy_score(y_test_project, y_pred_project)
        precision = precision_score(y_test_project, y_pred_project)
        f1 = f1_score(y_test_project, y_pred_project)
    except:
        pass


    project_test.append(project_name)
    scores_accuracy.append(accuracy)
    scores_precision.append(precision)
    scores_f1.append(f1)

    # Calculate the average accuracy score across all projects and folds
average_score_accuracy = sum(scores_accuracy) / len(scores_accuracy)
average_score_precision = sum(scores_precision) / len(scores_precision)
average_score_f1 = sum(scores_f1) / len(scores_f1)

# print("Average accuracy score:", average_score_accuracy)
# print("Average precision score:", average_score_precision)
project_test_name = []

for integer in project_test:
    string = dict_1.get(integer, None)
    project_test_name.append(string)

print(project_test_name)
print(scores_accuracy)
print(scores_precision)

# Creare un nuovo file CSV e aprire un writer
with open('/Users/valeriapontillo/Desktop/performance.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Scrivere l'intestazione del file
    writer.writerow(['projectName', 'Accuracy', 'Precision', 'F1-Measure'])

    # Scrivere i dati per ogni riga
    for i in range(len(project_test_name)):
        writer.writerow([project_test_name[i], scores_accuracy[i], scores_precision[i], scores_f1[i]])

    writer.writerow(["mean:", average_score_accuracy, average_score_precision, average_score_f1])

    '''i=0

    if not params["balancing"] == "none":
        print("Round " + str(i + 1) + " of " + str(k) + ": " + "Data balancing")
        X_train_project, y_train_project = data_balancing(params["balancing"], X_train_project, y_train_project)

    # classifier
    if not params["classifier"] == "none":
        clf_name = params["classifier"]
    else:
        clf_name = "dummy_random"
    clf = get_clf(clf_name)

    # hyperparameter opt

    if not params["optimization"] == "none" and not clf_name.startswith("dummy"):
        print("Round " + str(i + 1) + " of " + str(k) + ": " + "Hyperparameters optimization")
        best_params = hyperparam_opt(clf, clf_name, params["optimization"], X_train_project, y_train_project)
        save_best_params(best_params, out_dir + "best_params/" + str(i))
        clf.set_params(**best_params)

    # validation
    print("Round " + str(i + 1) + " of " + str(k) + ": " + "Training")
    clf.fit(X_train_project, y_train_project)

    del X_train_project
    del y_train_project
    gc.collect()

    print("Round " + str(i + 1) + " of " + str(k) + ": " + "Testing")
    # fpr, tpr, res, y_pred = scorer(clf, clf_name, X_test, y_test)
    precisionNew, recallNew, res, y_pred = scorer(clf, clf_name, X_test_project, y_test_project)
    y_pred = pd.DataFrame(y_pred, columns=["prediction"], index=X_test_project.index)
    y_pred.replace({0.0: False, 1.0: True}, inplace=True)

    #agreement = X_test_project.join(y_pred)
    #mode = 'w' if print_header else 'a'
    #agreement.to_csv(out_dir + "resultForTestCase.csv", mode=mode, header=print_header)
    #print_header = False
   # pyplot.figure()
    #pyplot.plot(recallNew, precisionNew)
    #pyplot.savefig(out_dir + "roc_curves/" + str(i) + ".png")
    precision_list.append(precisionNew)
    recall_list.append(recallNew)
    # fpr_list.append(fpr)
    # tpr_list.append(tpr)
    perf_df = pd.concat([perf_df, res], ignore_index=True)

    del X_test_project
    del y_test_project
    gc.collect()

    i = i + 1

print("Saving performance")

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
# perf_df  = pd.read_csv('performance.csv')
perf_df = perf_df.append(pd.Series(list, index=perf_df.columns[:len(list)]), ignore_index=True)
perf_df.to_csv(out_dir + "performance.csv")

'''
'''
    # Train the Decision Tree on the training data for this project
    clf.fit(X_train_project, y_train_project)

    # Make predictions on the testing data for this project
    try:
        y_pred_project = clf.predict(X_test_project)
    # Calculate the accuracy score for this project and add it to the scores list
        accuracy = accuracy_score(y_test_project, y_pred_project)
        precision = precision_score(y_test_project, y_pred_project)
        f1 = f1_score(y_test_project, y_pred_project)
    except:
        pass


    project_test.append(project_name)
    scores_accuracy.append(accuracy)
    scores_precision.append(precision)
    scores_f1.append(f1)

    # Calculate the average accuracy score across all projects and folds
average_score_accuracy = sum(scores_accuracy) / len(scores_accuracy)
average_score_precision = sum(scores_precision) / len(scores_precision)
average_score_f1 = sum(scores_f1) / len(scores_f1)

# print("Average accuracy score:", average_score_accuracy)
# print("Average precision score:", average_score_precision)
project_test_name = []

for integer in project_test:
    string = dict_1.get(integer, None)
    project_test_name.append(string)

print(project_test_name)
print(scores_accuracy)
print(scores_precision)

# Creare un nuovo file CSV e aprire un writer
with open('/Users/valeriapontillo/Desktop/performance.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Scrivere l'intestazione del file
    writer.writerow(['projectName', 'Accuracy', 'Precision', 'F1-Measure'])

    # Scrivere i dati per ogni riga
    for i in range(len(project_test_name)):
        writer.writerow([project_test_name[i], scores_accuracy[i], scores_precision[i], scores_f1[i]])

    writer.writerow(["mean:", average_score_accuracy, average_score_precision, average_score_f1])'''
