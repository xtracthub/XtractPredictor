
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, hamming_loss, multilabel_confusion_matrix, roc_auc_score
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss, ndcg_score
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def score_model(model, X_test, Y_test, class_table, multilabel):
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

    run_roc_auc = False
    if multilabel:
        run_roc_auc = True
        for i in range(len(y_pred[0])):
            if all(row[i] == 0 for row in y_pred):
                print(f"Label {i} was never predicted, roc_auc_score will not be used as an accuracy test.")
                run_roc_auc = False

    avg_prec_score_micro = precision_score(y_pred, Y_test, average='micro')
    avg_prec_score_macro = precision_score(y_pred, Y_test, average='macro')
    avg_prec_score_weighted = precision_score(y_pred, Y_test, average='weighted')

    avg_recall_score_micro = recall_score(y_pred, Y_test, average='micro')
    avg_recall_score_macro = recall_score(y_pred, Y_test, average='macro')
    avg_recall_score_weighted = recall_score(y_pred, Y_test, average='weighted')

    if multilabel:

        avg_prec_score_overall = precision_score(y_pred, Y_test, average='samples')
        avg_recall_score_overall = recall_score(y_pred, Y_test, average='samples')

        ml_accuracy_score = accuracy_score(y_pred, Y_test)
        print(f"\nModel accuracy (accuracy_score): {ml_accuracy_score}")

        if run_roc_auc:
            ml_roc_auc_score = roc_auc_score(y_pred, Y_test)
            print(f"Model accuracy (ROC AUC): {ml_roc_auc_score}")
            # accuracy = (ml_accuracy_score + ml_roc_auc_score) / 2
        # else:
        #     accuracy = ml_accuracy_score

        accuracy = ml_accuracy_score

        # UNCOMMENT TO INCLUDE HAMMING LOSS AS A METRIC
        # ml_hamming_gain = 1-hamming_loss(y_pred, Y_test)
        # print(f"Model accuracy (1-hamming loss): {ml_hamming_gain}")

        # UNCOMMENT TO MAKE MULTILABEL CONFUSION MATRIX
        # ml_multilabel_confusion_matrix = multilabel_confusion_matrix(y_pred, Y_test)
        # print(f'multilabel_confusion_matrix: {ml_multilabel_confusion_matrix}')

        ml_coverage_error = coverage_error(y_pred, Y_test)
        ml_label_ranking_average_precision_score = label_ranking_average_precision_score(y_pred, Y_test)
        ml_label_ranking_loss = label_ranking_loss(y_pred, Y_test)
        ml_ndcg_score = ndcg_score(y_pred, Y_test)

        print(f"Model precision (samples): {avg_prec_score_overall}")

        print(f"Model recall (samples): {avg_recall_score_overall}")

        print("\n---MULTILABEL RANKING METRICS---")
        print(f"Coverage error: {ml_coverage_error}")
        print(f"Label ranking average precision: {ml_label_ranking_average_precision_score}")
        print(f"Ranking loss: {ml_label_ranking_loss}")
        print(f"Normalized Discounted Cumulative Gain: {ml_ndcg_score}")
        print("---------------------------------\n")

    else:
        avg_prec_score_overall = (avg_prec_score_micro + avg_prec_score_macro + avg_prec_score_weighted) / 3
        avg_recall_score_overall = (avg_recall_score_micro + avg_recall_score_macro + avg_recall_score_weighted) / 3

        accuracy = model.score(X_test, Y_test)

        print(f"\nModel accuracy: {accuracy}")

        print(f"Model precision (micro): {avg_prec_score_micro}")
        print(f"Model precision (macro): {avg_prec_score_macro}")
        print(f"Model precision (weighted): {avg_prec_score_weighted}")

        print(f"Model recall (micro): {avg_recall_score_micro}")
        print(f"Model recall (macro): {avg_recall_score_macro}")
        print(f"Model recall (weighted): {avg_recall_score_weighted}")

    f1_score = 2*avg_prec_score_overall*avg_recall_score_overall/(avg_prec_score_overall+avg_recall_score_overall)
    print(f"F1-score: {f1_score}\n")

    # UNCOMMENT TO PRODUCE CONFUSION MATRIX
    ''' 
    disp = plot_confusion_matrix(model, X_test, Y_test,
                            display_labels=list(class_table.keys()),
                            cmap=plt.cm.Blues,
                            normalize=None)
    disp.ax_.set_title('SVC Confusion Matrix')
    plt.savefig('SVC Confusion Matrix No Normalize.png', format='png')
    '''

    # UNCOMMENT TO MAKE ROC CURVE
    # plot_multiclass_roc(model, X_test, Y_test, n_classes=6, figsize=(16, 10))

    # UNCOMMENT TO MAKE PRECISION-RECALL CURVE
    # plot_pr_curve(model, X_test, Y_test, 6) 

    return accuracy, avg_prec_score_overall, avg_recall_score_overall


def plot_multiclass_roc(clf, x_test, y_test, n_classes, figsize=(17, 6)):
    y_score = clf.decision_function(x_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('RF Receiver operating characteristic')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.savefig('RF ROC Curve.png', format='png')
    plt.show()
    

def plot_pr_curve(clf, x_test, y_test, n_classes):
    y_score = clf.predict_proba(x_test)

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
