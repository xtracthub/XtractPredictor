from headbytes import HeadBytes

headByteFeatures = HeadBytes(2048)

with open("feature_test_files/calibrum.jpg", "rb") as open_test_file:
	features = headByteFeatures.get_feature(open_test_file)

first_byte = features[0]
encoding = headByteFeatures.translate(first_byte)

print(encoding)