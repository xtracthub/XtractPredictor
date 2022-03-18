
import numpy as np
import json
from random import shuffle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV

from sklearn.metrics import plot_confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay  # uncomment the corresponding snippets to use


class ModelTrainer(object):
    def __init__(self, reader, model_param_dict, class_table_path, multilabel,
                 classifier="svc", split=0.8):
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
        self.multilabel = multilabel

        data = [line for line in reader.data]

        # Converts string sample labels to multilabel-friendly binary list
        if multilabel:

            unique_labels = []
            for i in range(len(data)):
                unique_labels.append(data[i][3])

            unique_labels = list(set(unique_labels))
            self.unique_labels = unique_labels
            self.num_of_labels = len(unique_labels)

            for i in range(len(data)):
                x, y = reader.feature.translate(data[i])
                data[i][2] = x
                sparse_list = [0] * self.num_of_labels
                sparse_list[unique_labels.index(data[i][3])] = 1
                data[i][3] = sparse_list

            squashed_data = []

            j = 0
            while j < len(data)-1:
                k = 1
                squashed_list = data[j][3]
                while j+k < len(data) \
                        and (data[j][0] == data[j+k][0]) \
                        and (data[j][1] == data[j+k][1]):
                    squashed_list = [a + b for a, b in zip(squashed_list, data[j+k][3])]
                    k += 1
                data[j][3] = squashed_list
                squashed_data.append(data[j])
                j += k

            data = squashed_data

        # Puts the data in a different order. (in-place action)
        shuffle(data)

        # Split the data into train and test sets (where split% of the data are for train)
        split_index = int(split * len(data))
        train_data = data[:split_index]  # split% of data.
        test_data = data[split_index:]  # 100% - split% of data.

        # np.zeros: create empty 2D X numpy array (and 1D Y numpy array) for features.
        self.X_train = np.zeros((len(train_data), int(reader.feature.nfeatures)))
        self.X_test = np.zeros((len(test_data), int(reader.feature.nfeatures)))

        if multilabel:
            self.Y_train = np.zeros((len(train_data), self.num_of_labels))
            self.Y_test = np.zeros((len(test_data), self.num_of_labels))
        else:
            self.Y_train = np.zeros(len(train_data))
            self.Y_test = np.zeros(len(test_data))

        groups = [[train_data, self.X_train, self.Y_train],
                  [test_data, self.X_test, self.Y_test]]

        # Here we merge the features into the empty X_train, ..., Y_test objects created above
        # --> Do this for both the train and the test data.

        if multilabel:
            for group in groups:
                raw_data, X, Y = group
                for i in range(len(raw_data)):
                    X[i] = raw_data[i][2]
                    Y[i] = raw_data[i][3]
        else:
            for group in groups:
                raw_data, X, Y = group
                for i in range(len(raw_data)):
                    x, y = reader.feature.translate(raw_data[i])
                    X[i] = x
                    Y[i] = y

        with open(class_table_path, 'w') as class_table:
            json.dump(reader.feature.class_table, class_table)

    def train(self):
        """Trains the model."""
        if self.classifier_type == "svc":
            self.model = SVC(kernel=self.params['kernel'], C=self.params['C'],
                             max_iter=self.params['iter'], degree=self.params['degree'])
        elif self.classifier_type == "logit":
            self.model = LogisticRegression(penalty=self.params['penalty'], solver=self.params['solver'],
                                            C=self.params['C'], max_iter=self.params['iter'], n_jobs=-1)
        elif self.classifier_type == "rf":
            self.model = RandomForestClassifier(n_estimators=self.params['n_estimators'],
                                                criterion=self.params['criterion'], max_depth=self.params['max_depth'],
                                                min_samples_split=self.params['min_sample_split'], n_jobs=-1)
        elif self.classifier_type == "dtc":
            self.model = DecisionTreeClassifier(criterion=self.params['criterion'],
                                                max_depth=self.params['max_depth'],
                                                min_samples_split=self.params['min_sample_split'],
                                                splitter=self.params['splitter'])
        elif self.classifier_type == "t_etc":  # TREE: EXTRA TREE CLASSIFIER
            self.model = ExtraTreeClassifier(criterion=self.params['criterion'],
                                             max_depth=self.params['max_depth'],
                                             min_samples_split=self.params['min_sample_split'],
                                             splitter=self.params['splitter'])
        elif self.classifier_type == "e_etc":  # ENSEMBLE: EXTRA TREES CLASSIFIER
            self.model = ExtraTreesClassifier(n_estimators=self.params['n_estimators'],
                                              criterion=self.params['criterion'],
                                              max_depth=self.params['max_depth'],
                                              min_samples_split=self.params['min_sample_split'],
                                              bootstrap=self.params['bootstrap'])
        elif self.classifier_type == "knc":
            self.model = KNeighborsClassifier(n_jobs=-1)
        elif self.classifier_type == "mlpc":
            print(f"solver: {self.params['solver']}")
            self.model = MLPClassifier()
        elif self.classifier_type == "rnc" and self.multilabel:
            self.model = RadiusNeighborsClassifier(outlier_label=[0]*self.num_of_labels,
                                                   n_jobs=-1,
                                                   leaf_size=self.params['leaf_size'],
                                                   algorithm=self.params['algorithm'])
        elif self.classifier_type == "rnc":
            self.model = RadiusNeighborsClassifier(n_jobs=-1)
        elif self.classifier_type == "rc":
            self.model = RidgeClassifier()
        elif self.classifier_type == "rccv":
            self.model = RidgeClassifierCV()

        self.model.fit(self.X_train, self.Y_train)

    def get_parameters(self):
        return self.params
    