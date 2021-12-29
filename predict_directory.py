import argparse
import pickle as pkl

import numpy as np
import json
import os
from features.headbytes import HeadBytes
from features.NaiveTruthReader import NaiveTruthReader 
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
        feature_obj = HeadBytes(head_size=head_bytes)
    elif feature == "randhead":
        feature_obj = RandHead(head_size=head_bytes, rand_size=rand_bytes)
    elif feature == "rand":
        feature_obj = RandBytes(number_bytes=rand_bytes)
    else:
        raise Exception("Not a valid feature set. ")


    # print(f"Features: {feature_obj}")
    # print(f"Filename: {filename}") 
    
    reader = NaiveTruthReader(feature_maker=feature_obj, labelfile='123')

    with open(filename, "rb") as open_file:
        features = reader.feature.get_feature(open_file)
        """
        row_data = ([os.path.dirname(row["path"]),
                     os.path.basename(row["path"]), 
                     features,
                     row["file_label"]])
        """
    # TODO: TYLER -- wtf is in reader.data.
    # reader = NaiveTruthReader(feature_maker=feature_obj, labelfile='cdiac_EAGLE.csv')
    # reader.run()
    # print(features)
    # exit()

    

    predict_start_time = time.time()
    extract_time = predict_start_time - start_extract_time

    # data = [line for line in reader.data][2]
    x = np.array([int.from_bytes(c, byteorder="big") for c in features])
    x = [x]

    prediction = trained_classifier.predict(x)
    prediction_probabilities = probability_dictionary(trained_classifier.predict_proba(x)[0], label_map)

    label = (list(label_map.keys())[list(label_map.values()).index(int(prediction[0]))])

    predict_time = time.time() - predict_start_time

    return label, prediction_probabilities, x, extract_time, predict_time


def probability_dictionary(probabilities, label_map):
    probability_dict = dict()
    for i in range(len(probabilities)):
        probability_dict[list(label_map.keys())[i]] = probabilities[i]
    return probability_dict


# filename = "/home/tskluzac/local_config.py"
# filename = "/home/tskluzac/.xtract/.test_files/animal-photography-olga-barantseva-11.jpg"
# filename = "/eagle/Xtract/cdiac/weather_2012.csv"
# filename = "/eagle/Xtract/cdiac/cdiac.ornl.gov/pub4/ndp055b/africa_forest_2000.gif"
# filename = "/eagle/Xtract/cdiac/cdiac.ornl.gov/pub8old/pub8/koertzinger_grl24.pdf"
# filename = "/eagle/Xtract/cdiac/cdiac.ornl.gov/pub8old/pub8/kaxis_underway_v3_201516030_60sec.csv"
# filename = "/eagle/Xtract/cdiac/cdiac.ornl.gov/pub8old/pub8/CdiacBundles/33GC/CO2_readme.doc"
# filename = "/eagle/Xtract/cdiac/cdiac.ornl.gov/pub8old/pub8/CdiacBundles/33GC/33GC20100914.tsv"
# filename = "/eagle/Xtract/cdiac/cdiac.ornl.gov/pub8old/pub8/oceans/Benjamin/For_Stew/CdiacBundles/33WA/WS1305-1_readme.xml"


"""
filename = "/eagle/Xtract/cdiac/cdiac.ornl.gov/pub8old/pub8/Hydrochem_data/in2016_v02Hydro012.nc"

trained_classifier = "stored_models/trained_classifiers/rf/rf-head-512-cdiac_EAGLE.csv-2021-12-23-23:01:06.pkl"
class_table_name = "stored_models/class_tables/rf/CLASS_TABLE-rf-head-512-cdiac_EAGLE.csv-2021-12-23-23:01:06.json"
feature = "head"
with open(trained_classifier, 'rb') as f:
    model = pkl.load(f)
x = predict_single_file(filename, model, class_table_name, feature)
print(x)
"""

def predict_directory(dir_name, trained_classifier, class_table_name, feature, head_bytes=512, rand_bytes=512):
    """
    Iterate over each file in a directory, and run a prediction for each file.
    :param dir_name:  (str) -- directory to be predicted
    :param trained_classifier:  (str) -- name of the classifier (from rf, svm, logit)
    :param feature: (str) -- from head, randhead, rand
    :param head_bytes: (int) the number of bytes to read from header (default: 512)
    :param rand_bytes: (int) the number of bytes to read from randomly throughout file
    """

    print(f"Directory name:\t{dir_name}")
    print(f"Trained Classifier:\t{trained_classifier}")
    print(f"Class Table:\t{class_table_name}")
    print(f"Feature:\t{feature}")


    file_predictions = dict()

    with open(trained_classifier, 'rb') as f:
        model = pkl.load(f)


    for subdir, dirs, files in os.walk(dir_name):
        for file_name in files:
            file_path = os.path.join(subdir, file_name)
            file_dict = dict()
            label, probabilities, _, _, _ = predict_single_file(file_path, model, class_table_name, feature, head_bytes, rand_bytes, should_print=False)
            file_dict['label'] = label
            file_dict['probabilities'] = probabilities
            file_predictions[file_path] = file_dict

    json.dump(file_predictions, open(dir_name + '_probability_predictions.json', 'w+'), indent=4)
    return file_predictions

"""
tc = "stored_models/trained_classifiers/rf/rf-head-512-cdiac_EAGLE.csv-2021-12-23-23:01:06.pkl"
class_table_name = "stored_models/class_tables/rf/CLASS_TABLE-rf-head-512-cdiac_EAGLE.csv-2021-12-23-23:01:06.json"

predict_directory(dir_name="/home/tskluzac/XtractPredictor", trained_classifier=tc, class_table_name=class_table_name, feature='head')
"""



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
    file_predictions = predict_directory(args.dir_name, args.trained_classifier, args.class_table,
                        args.feature, args.head_bytes, args.rand_bytes)

