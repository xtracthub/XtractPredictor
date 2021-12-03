import argparse
import time
import datetime
import pickle as pkl
import json
import os

from train.experiment import experiment
from features.headbytes import HeadBytes
from features.readers.readers import NaiveTruthReader
from classifiers.train_model import ModelTrainer
from classifiers.test_model import score_model
from features.randbytes import RandBytes
from features.randhead import RandHead
from classifiers.predict import predict_single_file, predict_directory


def extract_sampler(head_bytes, split=0.8, label_csv=None, dirname=None, predict_file=None,
                    trained_classifier=None, results_file="sampler_results.thing",
                    csv_outfile='naivetruth.csv', features_outfile=None, n_estimators=None,
                    criterion=None, max_depth=None, min_sample_split=None):

	features = HeadBytes(head_size=head_bytes)
	current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
	model_name = f"stored_models/trained_classifiers/{classifier}-{current_time}.pkl"

	reader = NaiveTruthReader(features, labelfile=label_csv)

	experiment(reader, split, features_outfile, 
				n_estimators, criterion, max_depth, 
				min_sample_split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run file classification experiments')

    parser.add_argument("--dirname", type=str, help="directory of files to predict if mode is predict"
                                                    "or directory to get labels and features of", default=None)
    parser.add_argument("--split", type=float, default=0.8,
                        help="test/train split ratio", dest="split")
    parser.add_argument("--head_bytes", type=int, default=512,
                        dest="head_bytes",
                        help="size of file head in bytes, default 512")
    parser.add_argument("--predict_file", type=str, default=None,
                        help="file to predict based on a classifier and a "
                             "feature")
    parser.add_argument("--trained_classifier", type=str,
                        help="trained classifier to predict on",
                        default='rf-head-default.pkl')
    parser.add_argument("--results_file", type=str,
                        default="sampler_results.json", help="Name for results file if predicting")
    parser.add_argument("--label_csv", type=str, help="Name of csv file with labels",
                        default="automated_training_results/naivetruth.csv")
    parser.add_argument("--csv_outfile", type=str, help="file to write labels to",
                        default='naivetruth.csv')
    parser.add_argument("--features_outfile", type=str, help="file to write features to if mode is labels_feautres"
                                                             "else it's a pkl with a reader object",
                        default=None)

    parser.add_argument("--n_estimators", type=int, help="sklearn Random Forest number of estimators (ignored in SVC and Logit)", default=30)
    parser.add_argument("--criterion", type=str, help="sklearn Random Forest criterion (ignored in SVC and Logit)", default='gini')
    parser.add_argument("--max_depth", type=int, help="sklearn Random Forest max_depth (ignored in SVC and Logit)", default=4000)
    parser.add_argument("--min_sample_split", type=int, help="sklearn Random Forest min_sample_split (ignored in SVC and Logit)", default=30)

    args = parser.parse_args()

    mdata = extract_sampler(args.classifier, args.head_bytes, args.split, args.label_csv, args.dirname,
							args.predict_file, args.trained_classifier, args.results_file, args.csv_outfile,
							args.features_outfile, args.n_estimators, args.criterion, args.max_depth,
							args.min_sample_split)