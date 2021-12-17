import os
import csv
import multiprocessing as mp
import pickle

class NaiveTruthReader(object):
    """Takes a .csv file of filepaths and file labels and returns a
    list of file directories, filepaths, features, and file labels.
    """
    def __init__(self, feature_maker, labelfile, feature_outfile):
        """Initializes NaiveTruthReader class.

        Parameters:
        feature_maker (str): An instance of a FileFeature class to extract
        features with (HeadBytes, RandBytes, RandHead).
        labelfile (.csv file): .csv file containing filepaths and labels.

        feature_file (str): path of feature file so we only need to read once 
        and then if the feature file already exists we can just load the read bytes
        instead of reextracting the bytes every time we want to train a model
        
        Return:
        (list): List of filepaths, file names, features, and labels.
        """
        self.feature = feature_maker
        self.data = []
        self.labelfile = labelfile
        self.feature_file = feature_outfile

    def extract_row_data(self, row):
        try:
            with open(row["path"], "rb") as open_file:
                features = self.feature.get_feature(open_file)
                row_data = ([os.path.dirname(row["path"]),
                            os.path.basename(row["path"]), features,
                            row["file_label"]])
                return row_data
        except (FileNotFoundError, PermissionError):
            print("Could not open %s" % row["path"])

    def run(self):
        if os.path.isfile(self.feature_file):
            self.data = pickle.load(open(self.feature_file, "rb"))
            print("Loading saved feature file...")
        else:
            labelf = open(self.labelfile, "r")

            reader = csv.DictReader(labelf)
            pools = mp.Pool(processes=mp.cpu_count())
            self.data = pools.map(self.extract_row_data, reader)
            pools.close()
            pools.join()
            for idx, item in enumerate(self.data):
                self.data[idx] = item
            if len(self.data) == 0:
                print("ERROR: Could not open any file. Stopping.")
                exit()
            pickle.dump(self.data, open(self.feature_file, "wb"))

    def get_feature_maker(self):
        return self.feature


def get_extension(filename):
    """Retrieves the extension of a file.

    Parameter:
    filename (str): Name of file you want to get extension from.

    Return:
    (str): File extension of filename.
    """
    if "." not in filename:
        return "None"
    return filename[filename.rfind("."):]
