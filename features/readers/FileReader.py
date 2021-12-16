import os
import csv
import multiprocessing as mp


"""
This file contains a number of classes that read one file, multiple files, or a LIST of files (available locally), 
and has functions to create (and access) the following: 

--> a list of file directories, filepaths, features, and file labels
"""

class FileReader(object):
    """Takes a single file and turns it into features."""

    def __init__(self, filename, feature_maker):
        """Initializes FileReader class.

        Parameters:
        filename (str): Name of file to turn into features.
        feature_maker (class): An instance of the HeadBytes,
        RandBytes, RandHead class.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError("%s is not a valid file" % filename)

        self.filename = filename
        self.feature = feature_maker
        self.data = []

    def handle_file(self, filename):
        """Extract features from a file.

        Parameter:
        filename (str): Name of file to extract features from.

        Return:
        (list): List of features and file extension of filename.
        """
        try:
            with open(filename, "rb") as open_file:
                extension = get_extension(filename)
                features = self.feature.get_feature(open_file)
                #print("basename: ", os.path.basename(filename))
                return [os.path.basename(filename), filename, features, extension]

        except (FileNotFoundError, PermissionError):
            pass

    def run(self):
        self.data = self.handle_file(self.filename)