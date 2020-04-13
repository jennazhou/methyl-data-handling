import gc
import ctypes

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.decomposition import FastICA

from sklearn.feature_selection import SelectFromModel

from multiprocessing import Process, Manager


y_train_sampled = np.load('../../datasets/preprocessed/y_train_sampled.npy')
y_test = np.load('../../datasets/preprocessed/y_test.npy')
X_train_scaled = np.load('../../datasets/preprocessed/X_train_scaled.npy')
X_test_scaled = np.load('../../datasets/preprocessed/X_test_scaled.npy')

# print(y_train_sampled)
# print(X_train_scaled)
print ("Shape of final train and test sets:", X_train_scaled.shape, X_test_scaled.shape)
    
C_options = [0.001, 0.01, 0.1, 1, 100, 1000]
# n_components = [30, 40, 50, 60, 70]
n_components = [200, 250,280] #for calculating vairance 
cm_tp=[[0,0],[0,0]]

#-------------------------------------------------------
# ### PCA
for n in n_components:
    pca = PCA(n_components=n)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)
    print("Cur n_compo:", n)
    print("Total variance of dataset covered:", np.sum(pca.explained_variance_ratio_))
    

#-------------------------------------------------------

### UMAP
# n_neighbours = [3, 5, 10, 15, 20]
# min_dist = [0.1, 0.25, 0.4, 0.5]
# for n in n_components:
#     for n_nei in n_neighbours:
#         for d in min_dist:
#             umap = UMAP(n_neighbors=n_nei, min_dist=d, n_components=n)
#             X_train = umap.fit_transform(X_train_scaled)
#             X_test = umap.transform(X_test_scaled)

#-------------------------------------------------------

# # ICA
# for n in n_components:
#     ica = FastICA(n_components=n)
#     X_train = ica.fit_transform(X_train_scaled)
#     X_test = ica.transform(X_test_scaled)

# # L1 Regularisation
# alpha=[0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
# for a in alpha:
#     sel_ = SelectFromModel(Lasso(alpha=a))
#     sel_.fit(X_train_scaled, y_train_sampled)
#     X_train = sel_.transform(X_train_scaled)
#     X_test = sel_.transform(X_test_scaled)
    
#     ####Use L1 LR for clf
#     grid = GridSearchCV(LogisticRegression(max_iter=500, penalty='l2', solver='saga'), param_grid=param_grid, scoring="accuracy", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=3)
#     grid.fit(X_train_selected, y_train)
#     mean_scores = np.array(grid.cv_results_['mean_test_score'])
#     print("FS using Regularisation with alpha=", a, "and l2:")
#     print(grid.cv_results_['params'])
#     print(mean_scores)
#     print(grid.best_params_)


    param_grid = [
        {
            'C': C_options,
        },
    ]

    print("Shape of X train:", X_train.shape)
    print("Shape of y train:", y_train_sampled.shape)

    grid = GridSearchCV(LogisticRegression(max_iter=1000, penalty='l2', solver='saga'), param_grid=param_grid, scoring="precision", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train_sampled)
    print("Mean score of precision of the best C:", grid.best_score_)
#     print("With Lasso(L2) FS alpha=",a,"and l2, the best params are:"

    #Use Testing set to check for overfitting
    clf = grid.best_estimator_
    y_pred_tr = clf.predict(X_train)
    y_pred_te = clf.predict(X_test)
    cm_tr = confusion_matrix(y_train_sampled, y_pred_tr)
    cm_te = confusion_matrix(y_test, y_pred_te)
#             print("Confusion matrix of PPMI training set:")
#             print(cm_tr)
#             print("Confusion matrix of PPMI testing set:")
#             print(cm_te)
    cur_params = {
#         "n_neighbour": n_nei,
#         "min_dist": d,
        "n_component":n
    }


#     if cm_te[0][0] >= 10:
    print("Confusion matrix of PPMI testing set:")
    print(cm_te)
    print("precision of testing set:", precision_score(y_test, y_pred_te))
    print(cur_params)
    print(grid.best_params_)
    print()

#     if cm_te[0][0] > cm_tp[0][0]:
#     cm_tp = cm_te
#     params_flag = cur_params
#     elif cm_te[0][0] == cm_tp[0][0] and cm_te[1][1] > cm_tp[1][1]:
#         cm_tp = cm_te
#         params_flag = cur_params

# print("UMAP + LR best performance params:")
# print("PCA + LR best performance params:")
# print(params_flag)
# print(cm_tp)



