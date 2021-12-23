import argparse
import pickle

import numpy as np
import json
import os
from features.headbytes import HeadBytes
from features.readers.readers import FileReader 
from features.randbytes import RandBytes
from features.randhead import RandHead
from sklearn.metrics import precision_score, recall_score
import time

def predict_single_file(filename, trained_classifier, class_table_name, feature, head_bytes=512, rand_bytes=512, should_print=False):
    """Predicts the type of file.
    filename (str): Name of file to predict the type of.
    trained_classifier: (sklearn model): Trained model.
    feature (str): Type of feature that trained_classifier was trained on.
    """

    start_extract_time = time.time()
    if should_print:
        print(f"Filename: {filename}")
        #print(f"Class table path: {class_table_name}")
    with open(class_table_name, 'r') as f:
        label_map = json.load(f)
        f.close()
    if feature == "head":
        features = HeadBytes(head_size=head_bytes)
    elif feature == "randhead":
        features = RandHead(head_size=head_bytes, rand_size=rand_bytes)
    elif feature == "rand":
        features = RandBytes(number_bytes=rand_bytes)
    else:
        raise Exception("Not a valid feature set. ")

    reader = FileReader(feature_maker=features, filename=filename)
    reader.run()

    predict_start_time = time.time()
    extract_time = predict_start_time - start_extract_time

    data = [line for line in reader.data][2]
    x = np.array([int.from_bytes(c, byteorder="big") for c in data])
    x = [x]

    prediction = trained_classifier.predict(x)
    prediction_probabilities = probability_dictionary(trained_classifier.predict_proba(x)[0], label_map)

    label = (list(label_map.keys())[list(label_map.values()).index(int(prediction[0]))])

    predict_time = time.time() - predict_start_time

    return label, prediction_probabilities, x, extract_time, predict_time

def predict_directory(dir_name, trained_classifier, class_table_name, feature, head_bytes=512, rand_bytes=512):
    """
    Iterate over each file in a directory, and run a prediction for each file.
    :param dir_name:  (str) -- directory to be predicted
    :param trained_classifier:  (str) -- name of the classifier (from rf, svm, logit)
    :param feature: (str) -- from head, randhead, rand
    :param head_bytes: (int) the number of bytes to read from header (default: 512)
    :param rand_bytes: (int) the number of bytes to read from randomly throughout file
    """
    file_predictions = dict()

    for subdir, dirs, files in os.walk(dir_name):
        for file_name in files:
            file_path = os.path.join(subdir, file_name)
            file_dict = dict()
            label, probabilities, _ = predict_single_file(file_path, trained_classifier, class_table_name, feature, head_bytes, rand_bytes, should_print=False)
            file_dict['label'] = label
            file_dict['probabilities'] = probabilities
            file_predictions[file_path] = file_dict

    
    json.dump(file_predictions,open(dir_name + '_probability_predictions.json', 'w+'), indent=4)
    return file_predictions

def probability_dictionary(probabilities, label_map):
    probability_dict = dict()
    for i in range(len(probabilities)):
        probability_dict[list(label_map.keys())[i]] = probabilities[i]
    return probability_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run file classification experiments')
    parser.add_argument("--dir_name", type=str, help="Directory to make predictions one")
    parser.add_argument("--trained_classifier", type=str, help="Path to saved and trained sklearn classifier")
    parser.add_argument("--class_table", type=str, help="Path to class table associated with trained classifier")
    parser.add_argument("--feature", type=str, help="Feature to classify directory on (head, rand, randhead) its reccomended you use the same feature as the one used to train the classifier on")
    parser.add_argument("--head_bytes", type=int, help="Number of head bytes to extract from each file, reccomended to have same value you used to also train the classifier on",
                        default=0)
    parser.add_argument("--rand_bytes", type=int, help="Number of rand bytes to extract from each file, reccomended to have same value you used to also train the classifier on",
                        default=0)

    args = parser.parse_args()

    model = pickle.load(open(args.trained_classifier, "rb"))

    file_predictions = predict_directory(args.dir_name, model, args.class_table,
                        args.feature, args.head_bytes, args.rand_bytes)