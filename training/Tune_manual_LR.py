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
from sklearn.metrics import accuracy_score, confusion_matrix
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
n_components = [10,20,30,40,50,60,70]

#-------------------------------------------------------
# ### PCA
# for n in n_components:
#     pca = PCA(n_components=n)
#     X_train = pca.fit_transform(X_train_scaled)
#     X_test = pca.transform(X_test_scaled)
#     print("PCA reduced check") 

#-------------------------------------------------------

# ### UMAP

# for n in n_components:
#     umap = UMAP(n_components=n)
#     X_train = umap.fit_transform(X_train_scaled)
#     X_test = umap.transform(X_test_scaled)

#-------------------------------------------------------

# # ICA
# for n in n_components:
#     ica = FastICA(n_components=n)
#     X_train = ica.fit_transform(X_train_scaled)
#     X_test = ica.transform(X_test_scaled)

# L1 Regularisation
alpha=[0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
for a in alpha:
    sel_ = SelectFromModel(Lasso(alpha=a))
    sel_.fit(X_train_scaled, y_train_sampled)
    X_train = sel_.transform(X_train_scaled)
    X_test = sel_.transform(X_test_scaled)
    
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

    grid = GridSearchCV(LogisticRegression(max_iter=1000, penalty='l2', solver='saga'), param_grid=param_grid, scoring="precision", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=-1, verbose=3)
    grid.fit(X_train, y_train_sampled)
    print("Mean score of precision of the best C:", grid.best_score_)
    print("With Lasso(L2) FS alpha=",a,"and l2, the best params are:")
    print(grid.best_params_, "for alpha=", a)

    #Use Testing set to check for overfitting
    clf = grid.best_estimator_
    print(clf)

    y_pred_tr = clf.predict(X_train)
    y_pred_te = clf.predict(X_test)
    cm_tr = confusion_matrix(y_train_sampled, y_pred_tr)
    cm_te = confusion_matrix(y_test, y_pred_te)
    print("Confusion matrix of PPMI training set:")
    print(cm_tr)
    print("Confusion matrix of PPMI testing set:")
    print(cm_te)
   




