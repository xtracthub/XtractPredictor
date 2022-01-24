

from argparse import ArgumentParser
from test_model import plot_multiclass_roc, plot_pr_curve




def plot_roc_and_pr(model, pkl_file):

    with open(model, 'rb') as f1:
        clf = pkl.load(f1)

    with open(pkl_file, 'rb') as f2:
        data = pkl.load(f2)

    x_test = data['x_test']
    y_test = data['y_test']
    
    print("Plotting multiclass ROC...")
    plot_multiclass_roc(model, x_test, y_test)



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--pkl_file')
    parser.add_argument('--model')
    args = parser.parse_args()


    plot_roc_and_pr(args.model, args.pkl_file)
