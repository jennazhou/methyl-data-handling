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
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.decomposition import FastICA
import xgboost as xgb

from sklearn.feature_selection import SelectFromModel

from multiprocessing import Process, Manager

from sklearn.cluster import KMeans

import joblib

path_to_npys = "/local/sdc/rz296/datasets/preprocessed/clustering/"
y = np.load(path_to_npys+'y.npy')
X_scaled = np.load(path_to_npys+'X_scaled.npy')

## clustering
X_for_clustering = X_scaled.transpose()
print("Shape of X for clustering:",  X_for_clustering.shape)
### Need to tune n_clusters
cluster_dict={}
# for k in ks:
k = 6
kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=3, verbose=1).fit(X_for_clustering)
clusters = kmeans.labels_
for n in range(k):
    print("Index of features in cluster", n+1, "is:")
    i = np.where(clusters == n)
    print(i)
    
    joblib.dump(i, "cluster_index_"+str(n)+".joblib") 
    
    print("Number of features in cluster",n+1, "=", i[0].size)
    X_cluster = X_for_clustering[i].transpose()
    print("Shape of X_train_cluster",n+1,"is", X_cluster.shape)
    cluster_dict[n] = X_cluster


#########
print("CLUSTERING INDEX SAVED TO FILE")
#########


## Apply fs
cluster_reduced_dict = {}
for i in range(k):
    sel_ = SelectFromModel(Lasso(alpha=0.08, tol=0.01, random_state=42))
    sel_.fit(cluster_dict[i], y)
    
    joblib.dump(sel_, "FS_selector_"+str(i)+".joblib") 
    
    X_reduced_cluster = sel_.transform(cluster_dict[i])
    cluster_reduced_dict[i] = X_reduced_cluster

temp = np.concatenate((cluster_reduced_dict[0], cluster_reduced_dict[1]), axis=1)
### Since only one cluster value used here, take X_reduced out
for j in range(3):
    temp =  np.concatenate((temp, cluster_reduced_dict[j+2]), axis=1)
X_reduced =  np.concatenate((temp, cluster_reduced_dict[k-1]), axis=1)
print("Final shape of input data after clustering:", X_reduced.shape)

######
print("ALL FS SELECTORS SAVED TO FILES")
#####

### Split training and testing sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split.get_n_splits(X_reduced, y)

for train_index, test_index in split.split(X_reduced, y):
    X_train, X_test = X_reduced[train_index], X_reduced[test_index]
    y_train, y_test= y[train_index], y[test_index]
    
### Upsample here
oversampler = SMOTE(random_state=42)
X_tr_sampled, y_tr_sampled = oversampler.fit_resample(X_train, y_train)



### Train on the concatenated clusters using XGBoost
clf = xgb.XGBClassifier(
            objective='binary:logistic', 
            seed=42, 
            tree_method='gpu_hist',
            learning_rate=0.5,
            subsample=0.3,
            gpu_id=1,
            colsample_bytree=0.4,
            n_estimators=70,
            max_depth=3
        )

clf.fit(X_tr_sampled, y_tr_sampled)

joblib.dump(clf, "GTB_model.joblib") 


######
print("GTB MODEL SAVED TO FILE")
#####


y_pred_tr = clf.predict(X_tr_sampled)
y_pred_te = clf.predict(X_test)
cm_tr = confusion_matrix(y_tr_sampled, y_pred_tr)
cm_te = confusion_matrix(y_test, y_pred_te)
print("Confusion matrix of clustered + reduced training set using UMAP+XGBoost:")
print(cm_tr)
print("Confusion matrix of clustered + reduced testing set using UMAP+XGBoost:")
print(cm_te)
print("precision of testing set:", precision_score(y_test, y_pred_te))
