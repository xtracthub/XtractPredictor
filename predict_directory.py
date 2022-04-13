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
from threading import Thread
from queue import Queue


class GetProbabilityVectors:
    def __init__(self, dir_name, trained_classifier, feature, head_bytes=512, rand_bytes=512):
        self.dir_name = dir_name
        self.trained_classifier = trained_classifier
        self.feature = feature
        self.head_bytes = head_bytes
        self.rand_bytes = rand_bytes

        self.files_q = Queue()
        self.preds_q = Queue()
        self.n_threads = 10

        with open(self.trained_classifier, 'rb') as f:
            self.classifier = pkl.load(f)

        self.load_queue(self.dir_name)
        print(f"Getting ready to predict {self.files_q.qsize()} files")

        threads = []
        for i in range(0, self.n_threads):
            thr = Thread(target=self.predict_single_thread, args=())
            thr.start()
            threads.append(thr)

        for running_thr in threads:
            running_thr.join()


        file_predictions = dict()
        while not self.preds_q.empty():
            file_dict = dict()
            file_path, label, probabilities, _, _, _ = self.preds_q.get()
            file_dict['label'] = label
            file_dict['probabilities'] = probabilities
            file_predictions[file_path] = file_dict

        json.dump(file_predictions, open(self.dir_name + '_probability_predictions.json', 'w+'), indent=4)
        

    def predict_single_thread(self):
        """Predicts the type of file.
        filename (str): Name of file to predict the type of.
        trained_classifier: (sklearn model): Trained model.
        feature (str): Type of feature that trained_classifier was trained on.
        """

        """
        with open(trained_classifier, 'rb') as f:
            trained_classifier = pkl.load(f)
        """

        while True: 
            #start_extract_time = time.time()
            #if should_print:
            #print(f"Filename: {filename}")
            
            print(f"Predictions completed: {self.preds_q.qsize()}")

            if self.files_q.empty():
                break

            filename = self.files_q.get()
            
            predict_start_time = time.time()
            label_map = self.classifier.class_table
            if self.feature == "head":
                feature_obj = HeadBytes(head_size=self.head_bytes)
            elif self.feature == "randhead":
                feature_obj = RandHead(head_size=self.head_bytes, rand_size=self.rand_bytes)
            elif self.feature == "rand":
                feature_obj = RandBytes(rand_bytes=self.rand_bytes)
            else:
                raise Exception("Not a valid feature set. ")
           
            t0 = time.time()
            reader = NaiveTruthReader(feature_maker=feature_obj, labelfile='123')

            # Tyler, you're going to read this and get upset but open() DOES NOT read entire file into memory.  
            with open(filename, "rb") as open_file:
                features = reader.feature.get_feature(open_file)

            t1 = time.time()
            #predict_start_time = time.time()
            #extract_time = predict_start_time - start_extract_time

            # data = [line for line in reader.data][2]
            x = np.array([int.from_bytes(c, byteorder="big") for c in features])
            x = [x]

            prediction = self.classifier.model.predict(x)
            prediction_probabilities = self.probability_dictionary(self.classifier.model.predict_proba(x)[0], label_map)

            label = (list(label_map.keys())[list(label_map.values()).index(int(prediction[0]))])

            predict_time = time.time() - predict_start_time
            extract_time = t1-t0

            writable = (filename, label, prediction_probabilities, x, extract_time, predict_time)
            self.preds_q.put(writable)


    def probability_dictionary(self, probabilities, label_map):
        probability_dict = dict()
        for i in range(len(probabilities)):
            probability_dict[list(label_map.keys())[i]] = probabilities[i]
        return probability_dict


    def load_queue(self, dir_name):
        """
        Iterate over each file in a directory, and run a prediction for each file.
        :param dir_name:  (str) -- directory to be predicted
        :param trained_classifier:  (str) -- name of the classifier (from rf, svm, logit)
        :param feature: (str) -- from head, randhead, rand
        :param head_bytes: (int) the number of bytes to read from header (default: 512)
        :param rand_bytes: (int) the number of bytes to read from randomly throughout file
        """

        #file_predictions = dict()

        #with open(trained_classifier, 'rb') as f:
        #    model = pkl.load(f)

        #num_predicted = 0
        for subdir, dirs, files in os.walk(dir_name):
            for file_name in files:
                file_path = os.path.join(subdir, file_name)
                self.files_q.put(file_path)
                #file_dict = dict()




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
    
    prob_class = GetProbabilityVectors(args.dir_name, args.trained_classifier,
                                       args.feature, args.head_bytes, args.rand_bytes)

