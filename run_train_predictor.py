import argparse
import time
import datetime
import pickle as pkl
import json
import os

from features.headbytes import HeadBytes
from features.readers.readers import NaiveTruthReader
from classifiers.train_model import ModelTrainer
from classifiers.test_model import score_model
from features.randbytes import RandBytes
from features.randhead import RandHead
from classifiers.predict import predict_single_file, predict_directory
# from automated_training import write_naive_truth
# from cloud_automated_training import write_naive_truth

# Global current time for saving models, class-tables, and training info.
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

def experiment(reader, classifier_name, features, 
               trials, split, model_name, features_outfile,
               model_param_dict):
    """Trains classifier_name on features from files in reader trials number
    of times and saves the model and returns training and testing data.

    Parameters:
    reader (list): List of file paths, features, and labels read from a
    label file.
    classifier_name (str): Type of classifier to use ("svc": support vector
    classifier, "logit": logistic regression, or "rf": random forest).
    features (str): Type of features to train on (head, rand, randhead,
    ngram, randngram).
    outfile (str): Name of file to write outputs to.
    trials (int): Number of times to train a model with randomized features
    for each training.
    split (float): Float between 0 and 1 which indicates how much data to
    use for training. The rest is used as a testing set.
    features_oufile (str): File path to .pkl file where a reader class is
    stored.

    Return:
    (pkl): Writes a pkl file containing the model.
    (json): Writes a json named outfile with training and testing data.
    """
    read_start_time = time.time()
    if features_outfile is None:
        reader.run()
    read_time = time.time() - read_start_time

    model_name = f"stored_models/trained_classifiers/{classifier_name}/{classifier_name}-{features}-{current_time}.pkl"
    class_table_path = f"stored_models/class_tables/{classifier_name}/CLASS_TABLE-{classifier_name}-{features}-{current_time}.json"
    classifier = ModelTrainer(reader, C, kernel, iter, degree, penalty, solver,
                              n_estimators, criterion, max_depth, min_sample_split,
                              class_table_path=class_table_path, classifier=classifier_name,
                              split=split)

    for i in range(trials):
        print("Starting trial {} out of {} for {} {}".format(i, trials,
                                                             classifier_name,
                                                             features))
        classifier_start = time.time()
        print("training")
        classifier.train()
        print("done training")
        accuracy, prec, recall = score_model(classifier.model, classifier.X_test,
                               classifier.Y_test, classifier.class_table)
        classifier_time = time.time() - classifier_start

        outfile_name = "{path}-info-{size}.json".format(path=os.path.splitext(model_name)[0], size=str(reader.get_feature_maker().get_number_of_features())+'Bytes')

        with open(model_name, "wb") as model_file:
            pkl.dump(classifier.model, model_file)
        with open(outfile_name, "a") as data_file:
            output_data = {"Classifier": classifier_name,
                           "Feature": features,
                           "Trial": i,
                           "Read time": read_time,
                           "Train and test time": classifier_time,
                           "Model accuracy": accuracy,
                           "Model precision": prec,
                           "Model recall": recall,
                           "Model size": os.path.getsize(model_name),
                           "Modifiable Parameters": classifier.get_parameters(),
                           "Parameters": classifier.model.get_params()}
            json.dump(output_data, data_file, indent=4)

        if i != trials-1:
            classifier.shuffle()

'''
Similar to extract sampler, except we're simplifying so that it only trains doesn't predict
'''
def train_extract_predictor(results_file, model_param_dict, classifier='rf',
                            feature='head', model_name=None, head_bytes=0, 
                            rand_bytes=0, split=0.8, label_csv=None, dirname=None,
                            predict_file=None, csv_outfile='naivetruth.csv'):

    if classifier not in ["svc", "logit", "rf"]:
        print("Invalid classifier option %s" % classifier)
        return
    if feature == "head":
        features = HeadBytes(head_size=head_bytes)
    elif feature == "rand":
        features = RandBytes(number_bytes=rand_bytes)
    elif feature == "randhead":
        features = RandHead(head_size=head_bytes,
                            rand_size=rand_bytes)
    else:
        print("Invalid feature option %s" % feature)
        return

    if model_name is None:
        model_name = f"stored_models/trained_classifiers/{classifier}-{feature}-{current_time}.pkl"

    reader = NaiveTruthReader(features, labelfile=label_csv)

    experiment(reader, classifier, feature, n,
                split, model_name, features_outfile, 
                model_param_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run file classification experiments')

    parser.add_argument("--dirname", type=str, help="directory of files to predict if mode is predict"
                                                    "or directory to get labels and features of", default=None)
    parser.add_argument("--classifier", type=str,
                        help="model to use: svc, logit, rf")
    parser.add_argument("--feature", type=str, default="head",
                        help="feature to use: head, rand, randhead")
    parser.add_argument("--split", type=float, default=0.8,
                        help="test/train split ratio", dest="split")
    parser.add_argument("--head_bytes", type=int, default=512,
                        dest="head_bytes",
                        help="size of file head in bytes, default 512")
    parser.add_argument("--rand_bytes", type=int, default=512,
                        dest="rand_bytes",
                        help="number of random bytes, default 512")
    parser.add_argument("--predict_file", type=str, default=None,
                        help="file to predict based on a classifier and a "
                             "feature")
    parser.add_argument("--results_file", type=str,
                        default="sampler_results.json", help="Name for results file if predicting")
    parser.add_argument("--label_csv", type=str, help="Name of csv file with labels",
                        default="automated_training_results/naivetruth.csv")
    parser.add_argument("--csv_outfile", type=str, help="file to write labels to",
                        default='naivetruth.csv')
    parser.add_argument("--model_name", type=str, help="Name of model",
                        default=None)

    parser.add_argument("--C", type=float, help="regularization parameter that is only useful in Logit and SVC", default=1)
    parser.add_argument("--kernel", type=str, help="Specified SVC Kernel (Ignored for others)", default='rbf')
    parser.add_argument("--iter", type=int, help="number of max iterations until it stops (relevant for SVC and Logit only)", default=-1)
    parser.add_argument("--degree", type=int, help="polynomial degree only for SVC when you specify poly kernel", default=3)

    parser.add_argument("--penalty", type=str, help="sklearn Logistic Regression penalty function (ignored for SVC and RF)", default='l2')
    parser.add_argument("--solver", type=str, help="sklearn Logistic Regression solver (ignored for SVC and RF)", default='lbfgs')

    parser.add_argument("--n_estimators", type=int, help="sklearn Random Forest number of estimators (ignored in SVC and Logit)", default=30)
    parser.add_argument("--criterion", type=str, help="sklearn Random Forest criterion (ignored in SVC and Logit)", default='gini')
    parser.add_argument("--max_depth", type=int, help="sklearn Random Forest max_depth (ignored in SVC and Logit)", default=4000)
    parser.add_argument("--min_sample_split", type=int, help="sklearn Random Forest min_sample_split (ignored in SVC and Logit)", default=30)

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
    
    mdata = train_extract_predictor(classifier=args.classifier, feature=args.feature, model_name=args.model_name,
                                    head_bytes=args.head_bytes, rand_bytes=args.rand_bytes, split=args.split, 
                                    csv_outfile=args.label_csv, dirname=args.dirname, predict_file=args.predict_file,
                                    results_file=args.results_file, model_param_dict=model_param_dict)