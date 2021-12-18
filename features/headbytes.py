import numpy as np
from features.feature import FeatureMaker
import sys

class HeadBytes(FeatureMaker):
    """Retrieves bytes from the head of a file."""
    def __init__(self, head_size):
        """Initializes HeadBytes class.

        Parameters:
        head_size (int): Number of bytes to get from header.
        """
        self.name = "head-" + str(head_size)
        self.head_size = head_size
        self.nfeatures = head_size
        self.class_table = {}

 
    def get_feature(self, open_file):
        """Retrieves the first head_size number of bytes from a file.

        Parameter:
        open_file (file): An opened file to retrieve data from.

        Return:
        head (list): A list of the first head_size bytes in
        open_file.
        If there are less than head_size bytes in
        open_file, the remainder of head is filled with empty bytes.
        """
        file_bytes = open_file.read(self.head_size) 
        head_bytes = [file_bytes[i:i+1] for i in range(0, len(file_bytes))]       
        if len(head_bytes) < self.head_size:
            head_bytes.extend([b'' for i in range(self.head_size - len(head_bytes))])
        assert len(head_bytes) == self.head_size
        return head_bytes

    def translate(self, entry):
        """Translates a feature into an integer.

        Parameter:
        entry (byte): A feature.

        Return:
        (tuple): 2-tuple of a numpy array containing an integer version of
        entry and a dictionary of labels and indices.
        """
        # a feature is a row with data on its file name, byte features, and label
        # entry[2] represents the entry with the byte features        

        x = [int.from_bytes(c, byteorder=sys.byteorder) for c in entry[2]]

        # y is the label for this vector of features 
        # last value of entry entry[-1] is the label so we get that label and we save it in y
        try:
            y = self.class_table[entry[-1]]
        except KeyError:
            # class table is filled out in the order that it is presented in the training data
            self.class_table[entry[-1]] = len(self.class_table) + 1
            y = self.class_table[entry[-1]]
        return np.array(x), y

