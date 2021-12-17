from NaiveTruthReader import NaiveTruthReader
from headbytes import HeadBytes
import numpy as np

feature_maker = HeadBytes(10)
reader = NaiveTruthReader(feature_maker, "test.csv")
reader.run()

data = [line for line in reader.data]

split_index = int(0.5 * len(data))
train_data = data[:split_index]  # split% of data.
test_data = data[split_index:]  # 100% - split% of data.

# np.zeros: create empty 2D X numpy array (and 1D Y numpy array) for features.
X_train = np.zeros((len(train_data), int(reader.feature.nfeatures)))
Y_train = np.zeros(len(train_data))

X_test = np.zeros((len(test_data), int(reader.feature.nfeatures)))
Y_test = np.zeros(len(test_data))

groups = [[train_data, X_train, Y_train],
        [test_data, X_test, Y_test]]


for group in groups:
	raw_data, X, Y = group
	print("new group")
	for i in range(len(raw_data)):
		print(raw_data[i][2])
		x, y = reader.feature.translate(raw_data[i])
		X[i] = x
		Y[i] = y
	print("X:", X)
	print("Y:", Y)