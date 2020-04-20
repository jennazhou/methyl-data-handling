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



## clustering
X_for_clustering = X_train_scaled.transpose()
print("Shape of X for clustering:",  X_for_clustering.shape)
### Need to tune n_clusters
ks =  [3]
cluster_reduced_dict={}
for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=k, verbose=1).fit(X_for_clustering)
    clusters = kmeans.labels_
    for n in range(k):
        print("Index of features in cluster", n+1, "is:")
        i = np.where(clusters == n)
        print(i)
        print("Number of features in cluster",n+1, "=", i[0].size)
        X_tr_cluster = X_for_clustering[i].transpose()
        print("Shape of X_train_cluster",n+1,"is", X_tr_cluster.shape)
        umap = UMAP(n_components=4)
        X_tr_umap = umap.fit_transform(X_tr_cluster)
        cluster_reduced_dict[n] = X_tr_umap

    print("Cluters dict:")
    print(cluster_reduced_dict)
    temp = np.concatenate(cluster_reduced_dict[0], cluster_reduced_dict[1])
    X_train =  np.concatenate(temp, cluster_reduced_dict[2])
    print("Final shape of input data after clustering:", X.shape)


### Train on the concatenated clusters
    clf = xgb.XGBClassifier(
                objective='binary:logistic', 
                seed=42, 
                tree_method='gpu_hist',
                learning_rate=0.3,
                subsample=0.7,
                gpu_id=1,
                colsample_bytree=0.5,
                n_estimators=10,
                max_depth=4
            )
    
    clf.fit(X_train, y_train_sampled)

    y_pred_tr = clf.predict(X_train)
    y_pred_te = clf.predict(X_test)
    cm_tr = confusion_matrix(y_train_sampled, y_pred_tr)
    cm_te = confusion_matrix(y_test, y_pred_te)
    print("Confusion matrix of clustered + reduced training set using UMAP+XGBoost:")
    print(cm_tr)
    print("Confusion matrix of clustered + reduced testing set using UMAP+XGBoost:")
    print(cm_te)
    print("precision of testing set:", precision_score(y_test, y_pred_te))

    