
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from sklearn.metrics import RocCurveDisplay

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from random import randint
import pickle as pkl

def score_model(model, X_test, Y_test, class_table):
    """Scores the model.

    Parameters:
    model (sklearn model): Trained sklearn model (SVC, LinearRegression,
    RandForestClassifier).
    X_test (list): List of files to test on.
    Y_test (list): List of labels for X_test.

    Return:
    (float): The percentage of files from X_test that model was able to
    correctly classify.
    """


    y_pred = model.predict(X_test)
    avg_prec_score_micro = precision_score(y_pred, Y_test, average='micro')
    avg_prec_score_macro = precision_score(y_pred, Y_test, average='macro')
    avg_prec_score_weighted = precision_score(y_pred, Y_test, average='weighted')
    avg_prec_score_overall = (avg_prec_score_micro + avg_prec_score_macro + avg_prec_score_weighted) / 3

    print(f"Model precision (micro): {avg_prec_score_micro}")
    print(f"Model precision (macro): {avg_prec_score_macro}")
    print(f"Model precision (weighted): {avg_prec_score_weighted}")

    avg_recall_score_micro = recall_score(y_pred, Y_test, average='micro')
    avg_recall_score_macro = recall_score(y_pred, Y_test, average='macro')
    avg_recall_score_weighted = recall_score(y_pred, Y_test, average='weighted')
    avg_recall_score_overall = (avg_recall_score_micro + avg_recall_score_macro + avg_recall_score_weighted ) / 3

    print(f"Model recall (micro): {avg_recall_score_micro}")
    print(f"Model recall (macro): {avg_recall_score_macro}")
    print(f"Model recall (weighted): {avg_recall_score_weighted}")

    accuracy = model.score(X_test, Y_test)
    print(f"Model accuracy: {accuracy}")

    # UNCOMMENT TO PRODUCE CONFUSION MATRIX
    """
    disp = plot_confusion_matrix(model, X_test, Y_test,
                            display_labels=list(class_table.keys()),
                            cmap=plt.cm.Blues,
                            normalize=None)
    disp.ax_.set_title('RF Confusion Matrix')
    plt.savefig('RF Confusion Matrix No Normalize.png', format='png')
    """
    # UNCOMMENT TO MAKE ROC CURVE
    print("Plotting multi-class ROC...")
    # plot_multiclass_roc(model, X_test, Y_test, n_classes=11, figsize=(16, 10))
    # rfc_disp = RocCurveDisplay.from_estimator(model, X_test, Y_test)




    #UNCOMMENT TO MAKE Precision Recall CURVE
    print("Plotting PR Curve...")
    # plot_pr_curve(model, X_test, Y_test, 11) 
    
    
    from random import randint
    with open(f"classifiers/stored_test_data/{randint(100000,999999)}.pkl", "wb") as f:
        savable ={'x_test': X_test, 'y_test': Y_test}
        pkl.dump(savable, f)
    

    return accuracy, avg_prec_score_overall, avg_recall_score_overall

def plot_multiclass_roc(clf, x_test, y_test, n_classes, figsize=(17, 6)):
   # SEE FOLLOWING LINK FOR ROC CURVE ON RANDOM FOREST
   # https://laurenliz22.github.io/roc_curve_multiclass_predictions_random_forest_classifier
    
    # y_score = clf.decision_function(x_test) # TYLER: UNCOMMENT THIS. 
    y_score = clf.predict_proba(x_test)

    y_test_bin = label_binarize(y_test, classes=range(1,11))

    n_classes = y_test_bin.shape[1]

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print(f"ROC AUC: {roc_auc}")
    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Fraction')
    ax.set_ylabel('True Positive Fraction')
    ax.set_title('RF')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.savefig('RF ROC Curve.png', format='png')
    # plt.show()
    

def plot_pr_curve(clf, x_test, y_test, n_classes):
    y_score = clf.predict_proba(x_test) # for rf make this predict_proba

    # precision recall curve
    precision = dict()
    recall = dict()


    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_dummies[:, i],
                                                        y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig('RF PR Curve.png', format='png')
    plt.show()
