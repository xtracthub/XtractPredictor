import argparse
import time
import datetime
import pickle as pkl
import json
import os

from classifiers.train_model import ModelTrainer
from classifiers.test_model import score_model

# Global current time for saving models, class-tables, and training info.
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S")


def experiment(reader, classifier_name, split, model_name, model_param_dict, multilabel):
    """Trains classifier_name on features from files in reader trials number
    of times and saves the model and returns training and testing data.

    Parameters:
    reader (list): List of file paths, features, and labels read from a
    label file.
    classifier_name (str): Type of classifier to use ("svc": support vector
    classifier, "logit": logistic regression, or "rf": random forest).
    split (float): Float between 0 and 1 which indicates how much data to
    use for training. The rest is used as a testing set.

    Return:
    (pkl): Writes a pkl file containing the model.
    (Sklearn Model): The copy of the model that was pkled
    """
    if model_name is None:
        model_name = f"stored_models/trained_classifiers/{classifier_name}/{classifier_name}-{reader.name}-{current_time}.pkl"
    else:
        model_name = f"stored_models/trained_classifiers/{classifier_name}/{model_name}-{reader.name}-{current_time}.pkl"

    class_table_path = f"stored_models/class_tables/{classifier_name}/CLASS_TABLE-{classifier_name}-{reader.name}-{current_time}.json"

    classifier = ModelTrainer(reader, model_param_dict, class_table_path, multilabel, classifier=classifier_name, split=split)

    classifier_start = time.time()
    print("training")
    classifier.train()
    print("done training")

    accuracy, prec, recall = score_model(classifier.model, classifier.X_test,
                                         classifier.Y_test, classifier.class_table,
                                         multilabel)

    classifier_time = time.time() - classifier_start

    outfile_name = "{path}-info-{size}.json".format(path=os.path.splitext(model_name)[0],
                                                    size=str(reader.get_feature_maker().get_number_of_features())
                                                         + 'Bytes')

    with open(model_name, "wb") as model_file:
        pkl.dump(classifier, model_file)
    with open(outfile_name, "a") as data_file:
        output_data = {"Classifier": classifier_name,
                       "Dataset Trained On": reader.name,
                       "Train and test time": classifier_time,
                       "Model accuracy": accuracy,
                       "Model precision": prec,
                       "Model recall": recall,
                       "Model size": os.path.getsize(model_name),
                       "Parameters": classifier.model.get_params(),
                       "Run as multilabel?": multilabel}
        if classifier_name == "e_etc":
            output_data["Bootstrap?"] = model_param_dict["bootstrap"]
        json.dump(output_data, data_file, indent=4)

    return classifier, outfile_name


'''
Similar to extract sampler, except we're simplifying so that it only trains doesn't predict
'''


def train_extract_predictor(model_param_dict, classifier, model_name, split, multilabel):
    if classifier not in ["svc", "logit", "rf", "dtc", "e_etc", "mlpc", "rccv", "t_etc", "knc", "rc", "rnc"]:
        print("Invalid classifier option %s" % classifier)
        return
    try:
        reader = pkl.load(open(args.feature_reader, "rb"))
    except OSError:
        print("ERROR: Feature reader file has not been specified/does not exist. Stopping.")
        return

    model, outfile_name = experiment(reader, classifier, split, model_name, model_param_dict, multilabel)
    return model, outfile_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run file classification experiments')
    parser.add_argument("--classifier", type=str,
                        help="model to use: svc, logit, rf", default="rf")
    parser.add_argument("--split", type=float, default=0.8,
                        help="test/train split ratio", dest="split")
    parser.add_argument("--model_name", type=str, help="Name of model",
                        default=None)
    parser.add_argument("--feature_reader", type=str, help="Path to Reader object that extracted directory's features",
                        default="")

    parser.add_argument("--C", type=float,
                        help="regularization parameter that is only useful in Logit and SVC", default=1)
    parser.add_argument("--kernel", type=str, help="Specified SVC Kernel (Ignored for others)", default='rbf')
    parser.add_argument("--iter", type=int,
                        help="number of max iterations until it stops (relevant for SVC and Logit only)", default=-1)
    parser.add_argument("--degree", type=int,
                        help="polynomial degree only for SVC when you specify poly kernel", default=3)

    parser.add_argument("--penalty", type=str,
                        help="sklearn Logistic Regression penalty function (ignored for SVC and RF)", default='l2')
    parser.add_argument("--solver", type=str,
                        help="sklearn Logistic Regression solver (ignored for SVC and RF)", default='lbfgs')

    parser.add_argument("--n_estimators", type=int,
                        help="sklearn Random Forest number of estimators (ignored in SVC and Logit)", default=30)
    parser.add_argument("--criterion", type=str,
                        help="sklearn Random Forest criterion (ignored in SVC and Logit)", default='gini')
    parser.add_argument("--max_depth", type=int,
                        help="sklearn Random Forest max_depth (ignored in SVC and Logit)", default=4000)
    parser.add_argument("--min_sample_split", type=int,
                        help="sklearn Random Forest min_sample_split (ignored in SVC and Logit)", default=30)

    parser.add_argument("--splitter", type=str,
                        help="Decision tree node splitting strategy", default="best")

    parser.add_argument("--algorithm", type=str,
                        help="sklearn Radius Neighbors Classifier neighbor computation algorithm", default="auto")
    parser.add_argument("--leaf_size", type=int,
                        help="sklearn Radius Neighbors Classifier arg passed to certain algorithms", default=30)

    parser.add_argument("--multilabel", dest="multilabel", action="store_true", default=False)
    parser.add_argument("--bootstrap", dest="bootstrap", action="store_true", default=False)

    args = parser.parse_args()

    model_param_dict = dict()
    model_param_dict["C"] = args.C
    model_param_dict["kernel"] = args.kernel
    model_param_dict["iter"] = args.iter
    model_param_dict["degree"] = args.degree
    model_param_dict["penalty"] = args.penalty
    model_param_dict["solver"] = args.solver
    model_param_dict["n_estimators"] = args.n_estimators
    model_param_dict["criterion"] = args.criterion
    model_param_dict["max_depth"] = args.max_depth
    model_param_dict["min_sample_split"] = args.min_sample_split
    model_param_dict["bootstrap"] = args.bootstrap
    model_param_dict["algorithm"] = args.algorithm
    model_param_dict["leaf_size"] = args.leaf_size
    model_param_dict["splitter"] = args.splitter

    model, outfile_name = train_extract_predictor(classifier=args.classifier, model_name=args.model_name,
                                                  split=args.split, model_param_dict=model_param_dict,
                                                  multilabel=args.multilabel)