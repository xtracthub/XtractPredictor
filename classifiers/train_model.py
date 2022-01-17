import numpy as np
import json
from random import shuffle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay # uncomment the corresponding snippets to use

class ModelTrainer(object):
    def __init__(self, reader, model_param_dict, class_table_path, classifier="svc", split=0.8):
        """Initializes the ModelTrainer class.
        reader (list): List of file paths, features, and labels read from a
        label file.
        model_param_dict (dict): Dictionary of parameters for model. If you fill in values 
        not associated with that classifier they are ignored. If you forget to fill in values
        default values are used (chosen when args were passed in on cmd)
        classifier (str): Type of classifier to use ("svc": support vector
        classifier, "logit": logistic regression, or "rf": random forest).
        split (float): Float between 0 and 1 which indicates how much data to
        use for training. The rest is used as a testing set.

        
        """
        self.classifier_type = classifier
        self.model = None
        self.class_table = reader.feature.class_table
        self.split = split
        self.params = model_param_dict
        
        data = [line for line in reader.data]

        # Puts the data in a different order. (in-place action)
        shuffle(data)

        # Split the data into train and test sets (where split% of the data are for train)
        split_index = int(split * len(data))
        train_data = data[:split_index]  # split% of data.
        test_data = data[split_index:]  # 100% - split% of data.

        # np.zeros: create empty 2D X numpy array (and 1D Y numpy array) for features.
        self.X_train = np.zeros((len(train_data), int(reader.feature.nfeatures)))
        self.Y_train = np.zeros(len(train_data))

        self.X_test = np.zeros((len(test_data), int(reader.feature.nfeatures)))
        self.Y_test = np.zeros(len(test_data))

        groups = [[train_data, self.X_train, self.Y_train],
                  [test_data, self.X_test, self.Y_test]]

        # Here we merge the features into the empty X_train, ..., Y_test objects created above
        # --> Do this for both the train and the test data.
        for group in groups:
            raw_data, X, Y = group
            for i in range(len(raw_data)):
                x, y = reader.feature.translate(raw_data[i])
                X[i] = x
                Y[i] = y

        # model_name = "{}-{}-{}.pkl".format(classifier, feature, timestamp)
        print(f"Train: Class table path: {class_table_path}")
        with open(class_table_path, 'w') as class_table:
            json.dump(reader.feature.class_table, class_table)

    def train(self):
        """Trains the model."""
        if self.classifier_type == "svc":
            self.model = SVC(kernel=self.params['kernel'], C=self.params['C'], # TODO: TYLER -- added probability=True here to fix issue in prediction. 
                             max_iter=self.params['iter'], degree=self.params['degree'], probability=True, verbose=True)
        elif self.classifier_type == "logit":
            self.model = LogisticRegression(penalty=self.params['penalty'], solver=self.params['solver'],
                                             C=self.params['C'], max_iter=self.params['iter'], n_jobs=-1)
        elif self.classifier_type == "rf":
            self.model = RandomForestClassifier(n_estimators=self.params['n_estimators'],
                                                 criterion=self.params['criterion'], max_depth=self.params['max_depth'], 
                                                 min_samples_split=self.params['min_sample_split'], n_jobs=-1)

        self.model.fit(self.X_train, self.Y_train)

    def get_parameters(self):
        return self.params
    
