import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

#Assert: dataset passed in is already ascaled
#Assert: clf passed in is already trained
c=0
def eval(clf_dict, X, y):
    for key in clf_dict:
        clf = clf_dict[key]
        print(clf)
        y_pred = clf.predict(X)
        cm = metrics.confusion_matrix(y, y_pred)
        acc = metrics.accuracy_score(y, y_pred)

        # Among all positives found, how many are true positives -> how precise is the prediction
        # percentage of your results which are relevant
        # care about percentage of the successful classification
        prec = metrics.precision_score(y, y_pred) 

        # How many positive found among all true positives
        # percentage of total relevant results correctly classified by your algorithm.
        # care about actuall positives
        recall = metrics.recall_score(y, y_pred) 

        # f1 combines both precision and recall, hence to achieve a high f1, both need to be relatively high
        f1 = metrics.f1_score(y, y_pred)
# #         auc = metrics.roc_auc_score(y, y_pred)
#         if c == 0:
#             linestyle=":"
#             color="tab:blue"
#         elif c == 1:
#             linestyle = "-"
#             color="b"
#         else:
#             linestyle = "--"
#             color="c"
#         c = c+1
#         fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
#         plt.plot(fpr, tpr, linestyle=linestyle, linewidth=2, color=color, label=key)
        print('Current model: ', key)
        print("Confusion matrix of ", key, "is: ", cm)
        print("Accuracy of ", key, "is: ", acc)
        print("Precision of ", key, "is: ", prec)
        print("Recall of ", key, "is: ", recall)
        print("F1 of ", key, "is: ", f1)
#         print("ROC AUC of ", clf, "is: ", auc)
        print()
    
    #Plot confusion matrix
    
#     plt.plot([0, 1], [0, 1], "k--")
#     plt.axis([0, 1, 0, 1.01]) #[xmin, xmax, ymin, ymax]
#     plt.xlabel("False positive rate (fpr)")
#     plt.ylabel("True positive rate (tpr)")
#     plt.legend(loc="lower right")
#     plt.show
        
