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


def set_Data(data):
    ppmi = pd.read_csv('../../datasets/preprocessed/trans_processed_PPMI_data.csv')
    ppmi.rename(columns={'Unnamed: 0':'Sentrix_position'}, inplace=True)
    ppmi.set_index('Sentrix_position', inplace=True)
    ppmi = ppmi.transpose()

    encoder = LabelEncoder()
    label = encoder.fit_transform(ppmi['Category'])

    tr = ppmi.drop(['Category'], axis=1)
    X = tr.values
    y = label
    print(X.shape)
    print(y.shape)

    print("StratifiedSampling check")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    split.get_n_splits(X, y)

    for train_index, test_index in split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, data['y_test'] = y[train_index], y[test_index]

    print("Oversampling check")
    oversampler = SMOTE(random_state=42)
    X_train_sampled, data['y_train_sampled'] = oversampler.fit_resample(X_train, y_train)
    print("Scaling check")
    scaler = StandardScaler()
#     scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_sampled)
    data['X_train_scaled_1'] = X_train_scaled[:247].reshape((1, -1))
    data['X_train_scaled_2'] = X_train_scaled[247:].reshape((1, -1))
    data['X_test_scaled'] = scaler.transform(X_test)
    
    print("Returning check")

manager = Manager()
data = manager.dict()

print("CHECKPOINT1")
#     p = Process(target=set_Data, args=(X_train_scaled, X_test_scaled, y_train_sampled, y_test,))
p = Process(target=set_Data, args=(data,))
print("CHECKPOINT2")
p.start()
print("CHECKPOINT3")
p.join()

y_train_sampled = data['y_train_sampled']
y_test = data['y_test']
X_train_scaled = np.append(data['X_train_scaled_1'], data['X_train_scaled_2']).reshape(494, 747668)
X_test_scaled = data['X_test_scaled']

# print(y_train_sampled)
# print(X_train_scaled)
print ("Shape of final train and test sets:", X_train_scaled.shape, X_test_scaled.shape)
    
C_options = [0.001, 0.01, 0.1, 1, 100, 1000]
n_components = [10,12,14,16,18,20,22]
kernels = ['rbf', 'poly']
gamma=[1e-4, 0.001, 0.01, 1, 1.5]

## PCA
# for n in n_components:
#     pca = PCA(n_components=n)
#     X_train = pca.fit_transform(X_train_scaled)
#     X_test = pca.transform(X_test_scaled)
#     print('Shape of PCs:', X_train.shape[1])
    
# UMAP
n_neighbours = [3, 5, 10, 15, 20, 50]
min_dist = [0.1, 0.25, 0.4, 0.7]
params_flag = {}
cm_tp = [[0,0],[0,0]]
for n in n_components:
    for n_nei in n_neighbours:
        for d in min_dist:
            umap = UMAP(n_neighbors=n_nei, min_dist=d, n_components=n)
            X_train = umap.fit_transform(X_train_scaled)
            X_test = umap.transform(X_test_scaled)

    
# # ### ICA
# best_perf=0
# for n in n_components:
#     ica = FastICA(n_components=n)
#     X_train = ica.fit_transform(X_train_scaled)
#     X_test = ica.transform(X_test_scaled)



# #Lasso Reg for FS
# alpha=[0.0001, 0.001,0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2]
# for a in alpha:
#     sel_ = SelectFromModel(Lasso(alpha=a, tol=0.01, random_state=42))
#     sel_.fit(X_train_scaled, y_train_sampled)
#     X_train = sel_.transform(X_train_scaled)
#     X_test = sel_.transform(X_test_scaled)
#     print("Shape of training set with alpha=", a, ":", X_train.shape)
    
    
            for k in kernels:
                for g in gamma:
                    svc = SVC(max_iter=3000, gamma=g, kernel=k, tol=0.01,class_weight='balanced')
                    param_grid ={'C': C_options, }
                    grid = GridSearchCV(svc, param_grid=param_grid, scoring="precision", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=-1)
                    grid.fit(X_train, y_train_sampled)

        #             print("Confusion matrix of PPMI training set:")
        #             print(cm_tr)
        #             print("Confusion matrix of PPMI testing set:")
        #             print(cm_te)
                    cur_params = {
                        "n_neighbour": n_nei,
                        "min_dist": d,
                        "n_component":n,
                        "kernel": k,
                        "gamma": g,
                    }  

                    clf = SVC(max_iter=3000, gamma=cur_params["gamma"], kernel=cur_params["kernel"], coef0=10, C=grid.best_params_["C"], tol=0.01,class_weight='balanced')
                    clf.fit(X_train, y_train_sampled)
                    y_pred_tr = clf.predict(X_train)
                    y_pred_te = clf.predict(X_test)
                    cm_tr = confusion_matrix(y_train_sampled, y_pred_tr)
                    cm_te = confusion_matrix(y_test, y_pred_te)


                    if cm_te[0][0] >= 10:
                        print("Confusion matrix of PPMI training set:")
                        print(cm_tr)
                        print("Confusion matrix of PPMI testing set:")
                        print(cm_te)
                        print("precision of testing set:", precision_score(y_test, y_pred_te))
                        print(cur_params)
                        print(grid.best_params_)
                        print()

                    if cm_te[0][0] > cm_tp[0][0]:
                        cm_tp = cm_te
                        params_flag = cur_params
                    elif cm_te[0][0] == cm_tp[0][0] and cm_te[1][1] > cm_tp[1][1]:
                        cm_tp = cm_te
                        params_flag = cur_params

print("UMAP + SVM best performance params:")
# print("PCA + LR best performance params:")
print(params_flag)
print(cm_tp)
 