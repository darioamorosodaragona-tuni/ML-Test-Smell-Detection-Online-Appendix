import argparse
import os
from ml_preparation import feature_selection_available, data_balancing_available
from ml_classification import optimization_available, classifiers_available


def collect_available_choices():
    choices = {}
    choices["feature_sel"] = feature_selection_available
    choices["balancing"] = data_balancing_available
    choices["optimization"] = optimization_available
    choices["classifier"] = classifiers_available
    return choices


def help_msg():
    msg = "available choices for customization of the pipeline:\n" 
    choices = collect_available_choices()
    for key in choices.keys():
        msg = msg + key + ": "
        for c in choices[key]:
            msg = msg + c + ", "
        msg = msg + "\n"
    return msg


def get_params():
    argparser = argparse.ArgumentParser()
    argparser.description = help_msg()
    argparser.add_argument('-i', help='input dataset: tfidf, tf or bow', required=True)
    argparser.add_argument('-k', help='k for k-fold cross-validation', type=int, default=10)
    argparser.add_argument('-p', nargs='*', help='pipeline customization choices (can be listed in any order)')
    args = argparser.parse_args()
    # collect parameters from command line
    params = {}
    params["data"] = args.i
    params["k"] = args.k
    # initialize dictionary with default pipeline customization choices 
    choices = collect_available_choices()
    for key in choices.keys():
        params[key] = "none"
    if args.p is not None:
        # replace default with custom defined choices 
        for cust in args.p:
            for key in choices.keys():
                if cust in choices[key]:
                    params[key] = cust
    return params


def id_string(params):
    id_str = ""
    for k in params.keys():
        if not k == "dataset_dir" and not k == "out_dir" and not params[k] == "none":
            id_str = id_str + str(params[k]) + "_"
    return id_str


def out_dir(dir_id):
    res_dir = "../data/models_results"
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    res_dir = res_dir + dir_id + "/"
    os.makedirs(res_dir)
    os.makedirs(res_dir + "roc_curves/")
    os.makedirs(res_dir + "best_params/")
    os.makedirs(res_dir + "features/")
    os.makedirs(res_dir + "IG/")
    os.makedirs(res_dir + "pr_curves/")
    os.makedirs(res_dir + "resultTestCase/")
    os.makedirs(res_dir + "resultPerformance/")
    return res_dir
