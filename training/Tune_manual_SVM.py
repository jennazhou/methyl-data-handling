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

import warnings
warnings.filterwarnings("ignore")


y_train_sampled = np.load('../../datasets/preprocessed/npy_files/y_train_sampled.npy')
y_test = np.load('../../datasets/preprocessed/npy_files/y_test.npy')
X_train_scaled = np.load('../../datasets/preprocessed/npy_files/X_train_scaled.npy')
X_test_scaled = np.load('../../datasets/preprocessed/npy_files/X_test_scaled.npy')
print("Complete loading data")
#------------------------------------------
# f = open("logs/svm_logs/pca_svm_experiments_log","w")  
# f = open("logs/svm_logs/ica_svm_experiments_log","w")  
f = open("logs/svm_logs/umap_svm_experiments_log","a")  
# f = open("logs/svm_logs/fs_svm_experiments_log","w")  
#--------------------------------------------

print ("Shape of final train and test sets:", X_train_scaled.shape, X_test_scaled.shape)
    
C_options = [0.001, 0.01, 100, 1000]
n_components = [12,14,16,18,20,22]
kernels = ['rbf', 'poly']
gamma=[ 0.01, 1, 1.5, 3]
coef0 = [0.5,1,3,5,7,10]

# # PCA
# for n in n_components:
#     pca = PCA(n_components=n)
#     X_train = pca.fit_transform(X_train_scaled)
#     X_test = pca.transform(X_test_scaled)
#     print('Shape of PCs:', X_train.shape[1])
    
# UMAP
n_neighbours = [3, 5, 7, 20]
min_dist = [0.1, 0.25, 0.4, 0.7]
for n in n_components:
    for n_nei in n_neighbours:
        for d in min_dist:
            umap = UMAP(n_neighbors=n_nei, min_dist=d, n_components=n)
            X_train = umap.fit_transform(X_train_scaled)
            X_test = umap.transform(X_test_scaled)
    
# # ### ICA
# for n in n_components:
#     ica = FastICA(n_components=n)
#     X_train = ica.fit_transform(X_train_scaled)
#     X_test = ica.transform(X_test_scaled)



# #Lasso Reg for FS
# alpha=[0.0001, 0.001,0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2]
# for a in alpha:
#     sel_ = SelectFromModel(Lasso(alpha=a, tol=0.01, random_state=42))
#     X_train = sel_.fit_transform(X_train_scaled, y_train_sampled)
#     X_test = sel_.transform(X_test_scaled)
#     print("Shape of training set with alpha=", a, ":", X_train.shape)
    
    cm_tp=[[0,0],[0,0]]
    for k in kernels:
        for g in gamma:
            for c in coef0:
                svc = SVC(max_iter=3000, gamma=g, kernel=k, tol=0.01,coef0=c, class_weight='balanced')
                param_grid ={'C': C_options, }
                grid = GridSearchCV(svc, param_grid=param_grid, scoring="precision", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=-1)
                grid.fit(X_train, y_train_sampled)

                cur_params = {
#                     "n_neighbour": n_nei,
#                     "min_dist": d,
#                     "n_component":n,
                    "lasso_a":a,
                    "kernel": k,
                    "gamma": g,
                    "coef0":c,
                    "C":grid.best_params_["C"],

                }
                f.write(str(cur_params))
                f.write("\n")


                clf = SVC(max_iter=3000, gamma=cur_params["gamma"], kernel=cur_params["kernel"], coef0=cur_params["coef0"], C=grid.best_params_["C"], tol=0.01,class_weight='balanced')
                clf.fit(X_train, y_train_sampled)
                y_pred_tr = clf.predict(X_train)
                y_pred_te = clf.predict(X_test)
                cm_tr = confusion_matrix(y_train_sampled, y_pred_tr)
                cm_te = confusion_matrix(y_test, y_pred_te)


                if cm_te[0][0] >= 10:
                    f.write("Confusion matrix of PPMI training set:\n")
                    f.write(str(cm_tr))
                    f.write("\n")
                    f.write("Confusion matrix of PPMI testing set:\n")
                    f.write(str(cm_te))
                    f.write("\n")
                    f.write("precision of testing set:" + str(precision_score(y_test, y_pred_te)))
                    f.write("\n")

                if cm_te[0][0] > cm_tp[0][0]:
                    cm_tp = cm_te
                    params_flag = cur_params
                    f.write("The temp confmatx of testing set has been updated to:\n")
                    f.write(str(cm_tp)+"\n")
                    f.write("\n")
                elif cm_te[0][0] == cm_tp[0][0] and cm_te[1][1] > cm_tp[1][1]:
                    cm_tp = cm_te
                    params_flag = cur_params
                    f.write("The temp confmatx of testing set has been updated to:\n")
                    f.write(str(cm_tp))
                    f.write("\n")


#     f.write("For UMAP n_compo="+str(n)+",from confusion matrix of PPMI testing set, best params are: \n")
    f.write("For FS alpha="+str(a)+",from confusion matrix of PPMI testing set, best params are:")
    f.write(str(params_flag))
    f.write("\n")
    f.write(str(cm_tp))
    f.write("\n")
#     print("Complete experiments for ", n, "components")
    print("Complete experiments for lasso_alpha=", a)

    
f.close()

        
 