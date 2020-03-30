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

# C_options = [0.001, 0.01, 0.1, 1, 100, 1000]
# n_components = [3,5,7,9,11,13]
    
# # Train XGBoost classifier
# # DMatrix: a data structure that makes everything more efficient
# # dtrain = xgb.DMatrix(X_train_scaled, y_train)
# # dtest = xgb.DMatrix(X_test_scaled, y_test)


# # print('Already skimmed parameters')
# # num_round = 50
# # bst = xgb.train(param, dtrain, num_round, [(dtest, 'test'), (dtrain, 'train')])
# # # y_pred = bst.predict(dtest)
# # bst

# # ## PCA
# # for n in n_components:
# #     pca = PCA(n_components=n)
# #     X_train = pca.fit_transform(X_train_scaled)
# #     X_test = pca.transform(X_test_scaled)



# # ## UMAP
# # for n in n_components:
# #     umap = UMAP(n_components=n)
# #     X_train = umap.fit_transform(X_train_scaled)
# #     X_test = umap.transform(X_test_scaled)

# # # iCA
# # for n in n_components:
# #     ica = FastICA(n_components=n)
# #     X_train = ica.fit_transform(X_train_scaled)
# #     X_test = ica.transform(X_test_scaled)

# alpha=[0.007, 0.009, 0.015, 0.02, 0.04]
# for a in alpha:
#     sel_ = SelectFromModel(Lasso(alpha=a, tol=0.01, random_state=42))
#     sel_.fit(X_train_scaled, y_train_sampled)
#     X_train = sel_.transform(X_train_scaled)
#     X_test = sel_.transform(X_test_scaled)
#     print("Shape of training set with alpha=", a, ":", X_train.shape)
    
    
# #     param_grid = [
# #         {
# #             'C': C_options,
# #         },
# #     ]

# #     grid = GridSearchCV(LogisticRegression(max_iter=500, penalty='l1', solver='saga'), param_grid=param_grid, scoring="accuracy", cv=3, n_jobs=-1)
# #     grid.fit(X_train, y_train)
# #     # evaluation metric is accuray 
# #     mean_scores = np.array(grid.cv_results_['mean_test_score'])
# #     print("With PCA=",n,"and l1:")
# #     print(grid.cv_results_['params'])
# #     print(mean_scores)
# #     print(grid.best_params_)
    
# #     dtrain = xgb.DMatrix(X_train, y_train)
# #     dtest = xgb.DMatrix(X_test, y_test)
# #     param = {
# #         'objective' : 'binary:logistic', 
# #         'seed' : 42, 
# #         'tree_method':'gpu_hist'
# #         }

 
#     colsample_bytree = [0.1, 0.3, 0.5 , 0.7 ]
#     ne = [10, 30, 50, 70, 100, 150, 200, 300] 
#     subsample = [0.3, 0.5, 0.7]

#     best_perf=0
#     cm_tp=[[0,0],[0,0]]
#     for n_estimator in ne:
#         for colsample in colsample_bytree:
#             for ss in subsample:
#                 xgb_clf = xgb.XGBClassifier(
#                     objective='binary:logistic', 
#                     seed=42, 
#                     tree_method='gpu_hist',
#                     learning_rate=0.3,
#                     subsample=ss,
#                     gpu_id=1,
#                     colsample_bytree=colsample,
#                     n_estimators=n_estimator
#                 )

#                 param_grid = {
#                     "max_depth": [2,3,4]
#                    }

#                 grid = GridSearchCV(xgb_clf, param_grid=param_grid, scoring="precision", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), verbose=0)
#                 grid.fit(X_train, y_train_sampled)
#     #             print("Mean score of precision of the best max_depth:", grid.best_score_)
#     #             print()
#                 print("Current score:", grid.best_score_)
#                 cur_params = {
#                     "num_estmtr": n_estimator,
#                     "col_ratio": colsample,
#                     "subsample_ratio": ss,
#                     "max_depth": grid.best_params_["max_depth"]
#                 }
#                 clf_sub = grid.best_estimator_
# #                 y_pred_tr = clf_sub.predict(X_train)
#                 y_pred_te = clf_sub.predict(X_test)
# #                 cm_tr = confusion_matrix(y_train_sampled, y_pred_tr)
#                 cm_te = confusion_matrix(y_test, y_pred_te)
# #                 print("Confusion matrix of PPMI training set:")
# #                 print(cm_tr)
# #                 if cm_te[0][0] >= 10:
# #                     print("Confusion matrix of PPMI testing set:")
# #                     print(cm_te)
# #                     print("precision of testing set:", precision_score(y_test, y_pred_te))
# #                     print(cur_params)
# #                     print()
# #                 if grid.best_score_ > best_perf:
# #                     best_perf = grid.best_score_
# #                     best_param = grid.best_params_
# #                     tree_num_flag = n_estimator
# #                     col_ratio_flag = colsample
# #                     ss_flag = ss
#                 if cm_te[0][0] > cm_tp[0][0]:
#                     cm_tp = cm_te
#                     params_flag = cur_params
#                 elif cm_te[0][0] == cm_tp[0][0] and cm_te[1][1] > cm_tp[1][1]:
#                     cm_tp = cm_te
#                     params_flag = cur_params


# #     print("XGBoost with n_compo=", n, ', num_estmtr=', tree_num_flag,',col_ratio=',col_ratio_flag,'subsample=',ss_flag,'has best performance of',best_perf, "with", best_param)
# #     print("For ICA n_compo=",n,",from confusion matrix of PPMI testing set, best params are:")
#     print("For FS alpha=",a,",from confusion matrix of PPMI testing set, best params are:")
#     print(params_flag)
#     print(cm_tp)
#     print()
    
#     #Use Testing set to check for overfitting
#     clf = xgb.XGBClassifier(
#                 objective='binary:logistic', 
#                 seed=42, 
#                 tree_method='gpu_hist',
#                 learning_rate=0.3,
#                 subsample=params_flag["subsample_ratio"],
#                 gpu_id=1,
#                 colsample_bytree=params_flag["col_ratio"],
#                 n_estimators=params_flag["num_estmtr"],
#                 max_depth=params_flag["max_depth"]
#             )
#     print(clf)
#     clf.fit(X_train, y_train_sampled)

#     y_pred_tr = clf.predict(X_train)
#     y_pred_te = clf.predict(X_test)
#     cm_tr = confusion_matrix(y_train_sampled, y_pred_tr)
#     cm_te = confusion_matrix(y_test, y_pred_te)
#     print("Confusion matrix of PPMI training set:")
#     print(cm_tr)
#     print("Confusion matrix of PPMI testing set:")
#     print(cm_te)
#     print("precision of testing set:", precision_score(y_test, y_pred_te))
   
