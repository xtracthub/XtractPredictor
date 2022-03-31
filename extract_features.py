import argparse
import time
import datetime
import pickle as pkl
import json

from features.NaiveTruthReader import NaiveTruthReader
from features.headbytes import HeadBytes
from features.randbytes import RandBytes
from features.randhead import RandHead

# Global current time for labeling feature outfiles
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def extract_features(feature, head_bytes, rand_bytes, label_csv):
    if feature == "head":
        features = HeadBytes(head_size=head_bytes)
    elif feature == "rand":
        features = RandBytes(number_bytes=rand_bytes)
    elif feature == "randhead":
        features = RandHead(head_size=head_bytes, rand_size=rand_bytes)
    else:
        print("Invalid feature option %s" % feature)
        return
    
    feature_file = f"stored_features/{label_csv}-{current_time}-features.pkl"
    reader = NaiveTruthReader(features, labelfile=label_csv)
    
    read_start_time = time.time()
    reader.run()
    read_time = time.time() - read_start_time
    pkl.dump(reader, open(feature_file, "wb"))
    outfile_name = f"stored_features/{label_csv}-{current_time}-read-info.json"

    with open(outfile_name, "a") as data_file:
        output_data = {"Feature": feature,
                        "HeadBytes": head_bytes,
                        "RandBytes": rand_bytes,
                        "Read time(seconds)": read_time}
        json.dump(output_data, data_file, indent=4)

    return reader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract bytes (features) from files')
    parser.add_argument("--feature", type=str, default="head",
                        help="feature to use: head, rand, randhead")
    parser.add_argument("--head_bytes", type=int, default=0,
                        dest="head_bytes",
                        help="size of file head in bytes")
    parser.add_argument("--rand_bytes", type=int, default=0,
                        dest="rand_bytes",
                        help="number of random bytes")
    parser.add_argument("--label_csv", type=str, help="Name of csv file with labels", default=None)

    args = parser.parse_args()

    if args.label_csv == None:
        print("ERROR: No label csv specified")
        exit()
        
    reader = extract_features(feature=args.feature, head_bytes=args.head_bytes,
                              rand_bytes=args.rand_bytes, label_csv=args.label_csv)
