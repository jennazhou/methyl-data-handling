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

import joblib

from sklearn.feature_selection import SelectFromModel

from multiprocessing import Process, Manager

from sklearn.cluster import KMeans

# def set_Data(data):
#     ppmi = pd.read_csv('../../datasets/preprocessed/trans_processed_PPMI_data.csv')
#     ppmi.rename(columns={'Unnamed: 0':'Sentrix_position'}, inplace=True)
#     ppmi.set_index('Sentrix_position', inplace=True)
#     ppmi = ppmi.transpose()

#     encoder = LabelEncoder()
#     label = encoder.fit_transform(ppmi['Category'])

#     tr = ppmi.drop(['Category'], axis=1)
#     X = tr.values
#     y = label
#     print(X.shape)
#     print(y.shape)

# #     print("StratifiedSampling check")
# #     split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# #     split.get_n_splits(X, y)

# #     for train_index, test_index in split.split(X, y):
# #         X_train, X_test = X[train_index], X[test_index]
# #         y_train, data['y_test'] = y[train_index], y[test_index]

# #     print("Oversampling check")
# #     oversampler = SMOTE(random_state=42)
# #     X_sampled, data['y_sampled'] = oversampler.fit_resample(X, y)
# #     print(X_sampled.shape)
#     data['y'] = y
#     print("Scaling check")
#     scaler = StandardScaler()
# #     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
#     data['X_scaled_1'] = X_scaled[:300].reshape((1, -1))
#     data['X_scaled_2'] = X_scaled[300:].reshape((1, -1))
    
#     print("Returning check")

# manager = Manager()
# data = manager.dict()

# print("CHECKPOINT1")
# #     p = Process(target=set_Data, args=(X_train_scaled, X_test_scaled, y_train_sampled, y_test,))
# p = Process(target=set_Data, args=(data,))
# print("CHECKPOINT2")
# p.start()
# print("CHECKPOINT3")
# p.join()

# y = data['y']
# X_scaled = np.append(data['X_scaled_1'], data['X_scaled_2']).reshape(436, 747668)



####set params
n_jobs = 3



###Load data directly
X_scaled = np.load('../../datasets/preprocessed/clustering/X_scaled.npy')
y = np.load('../../datasets/preprocessed/clustering/y.npy')


# print ("Shape of final train and test sets:", X_train_scaled.shape, X_test_scaled.shape)    

## clustering
X_for_clustering = X_scaled.transpose()
# X_ppg_for_k = X_scaled_ppg.transpose()
print("Shape of X for clustering:",  X_for_clustering.shape)
### Need to tune n_clusters
cluster_dict={}
# cluster_dict_ppg = {}
k = 6
kmeans = KMeans(n_clusters=6, random_state=0, n_jobs=n_jobs, verbose=1).fit(X_for_clustering)
clusters = kmeans.labels_
for n in range(k):
    print("Index of features in cluster", n+1, "is:")
    i = np.where(clusters == n)
    print(i)
    print("Number of features in cluster",n+1, "=", i[0].size)
    X_cluster = X_for_clustering[i].transpose()
    print("Shape of X_train_cluster",n+1,"is", X_cluster.shape)
    cluster_dict[n] = X_cluster
    
    
    
    
cluster_reduced_dict = {}
for i in range(k):
    ###FS
    sel_ = SelectFromModel(Lasso(alpha=0.005, tol=0.01, random_state=42))
    sel_.fit(cluster_dict[i], y)
    
    ##Save weights of each cluster's FS
    filename = '../../trained-ml-models/clustering/FS'+str(i)+".joblib"
    joblib.dump(sel_, filename)
    print("FS for cluster ",str(i), "saved")
    
    X_reduced_cluster = sel_.transform(cluster_dict[i])
    cluster_reduced_dict[i] = X_reduced_cluster
print(cluster_reduced_dict)
temp = np.concatenate((cluster_reduced_dict[0], cluster_reduced_dict[1]), axis=1)
### Since only one cluster value used here, take X_reduced out


for j in range(3):
    temp =  np.concatenate((temp, cluster_reduced_dict[j+2]), axis=1)
X_reduced =  np.concatenate((temp, cluster_reduced_dict[k-1]), axis=1)
print("Final shape of input data after clustering:", X_reduced.shape)


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
            gpu_id=0,
            colsample_bytree=0.4,
            n_estimators=70,
            max_depth=3
        )

clf.fit(X_tr_sampled, y_tr_sampled)

###Save weights
filename = "../../trained-ml-models/clustering/gtb.joblib"
joblib.dump(clf, filename)
print("GTB weights saved")
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
