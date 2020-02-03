import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.decomposition import FastICA

from sklearn.feature_selection import SelectFromModel

from joblib import Parallel, delayed


ppmi = pd.read_csv('./trans_processed_PPMI_data.csv')
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

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split.get_n_splits(X, y)

for train_index, test_index in split.split(X, y):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    
    
C_options = [0.01, 0.1, 1, 100]

# for C in C_options:
#     split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     #split.get_n_splits(X, y)
#     #
#     #for train_index, test_index in split.split(X, y):
#     #    print("TRAIN:", len(train_index), "TEST:", len(test_index))
#     #    X_train, X_test = X[train_index], X[test_index]
#     #    y_train, y_test = y[train_index], y[test_index]
#     #
#     #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#     t_split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
#     t_split.get_n_splits(X, y)
#     for rem_index, tune_train_index in t_split.split(X, y):
#         print("TUNE TRAIN:", len(tune_train_index), "REMAINING:", len(rem_index))
#         X_tune_train, X_remaining = X[tune_train_index], X[rem_index]
#         y_tune_train, y_remaining = y[tune_train_index], y[rem_index]

#     split.get_n_splits(X_remaining, y_remaining)
#     for unused, tune_test_index in split.split(X_remaining, y_remaining):
#         print("TUNE TEST:", len(tune_test_index), "UNUSED:", len(unused))
#         X_tune_test = X_remaining[tune_test_index]
#         y_tune_test = y_remaining[tune_test_index]


#     scaler = StandardScaler()
#     #X_train_scaled = scaler.fit_transform(X_train)
#     #X_train_scaled
#     #X_test_scaled = scaler.fit_transform(X_test)
#     X_tune_train_scaled = scaler.fit_transform(X_tune_train)
#     X_tune_test_scaled = scaler.transform(X_tune_test)

#     #l1_ratio = [0.2, 0.5, 0.8]

#     #param_grid = [
#     #    {
#     #        'C': C_options,
#     #        'penalty': ['l1']
#     #    },
#     #    {
#     #        'C': C_options,
#     #        'penalty': ['l2']
#     #    },
#     #    {
#     #        'C': C_options,
#     #        'penalty': ['elsticnet'],
#     #        'l1_ratio': l1_ratio
#     #    }
#     #]
#     #
#     lr =LogisticRegression(max_iter=700, penalty='l1',solver='saga', C=C)
#     lr.fit(X_tune_train_scaled, y_tune_train)
#     y_tune_pred = lr.predict(X_tune_test_scaled)
#     print('Accuracy of LR with l1 with C value=', C, ': ', accuracy_score(y_tune_pred, y_tune_test))


# #########Use grid search properly###########
# # l1_ratio = [0.2, 0.5, 0.8]

# param_grid = [
#     {
#         'C': C_options,
#     }
# ]

# lr =  LogisticRegression(max_iter=500, penalty='l1', solver='saga')

# grid = GridSearchCV(lr, param_grid=param_grid, scoring="accuracy", cv=6, n_jobs=3)
# grid.fit(X_train_scaled, y_train)

# mean_scores = np.array(grid.cv_results_['mean_test_score'])
# print(mean_scores)
# print('Best estimator of C value with L1 for LR: ', grid.best_params_)

#-------------------------------------------------------
### PCA
# n_components = [50, 100, 150, 200, 250]

# for n in n_components:
#     pca = PCA(n_components=n)
#     X_train = pca.fit_transform(X_train_scaled)
#     X_test = pca.transform(X_test_scaled)

#     param_grid = [
#         {
#             'C': C_options,
#         },
#     ]

#     grid = GridSearchCV(LogisticRegression(max_iter=500, penalty='l1', solver='saga'), param_grid=param_grid, scoring="accuracy", cv=3, n_jobs=-1)
#     grid.fit(X_train, y_train)
#     # evaluation metric is accuray 
#     mean_scores = np.array(grid.cv_results_['mean_test_score'])
#     print("With PCA=",n,"and l1:")
#     print(grid.cv_results_['params'])
#     print(mean_scores)
#     print(grid.best_params_)

#-------------------------------------------------------

# ### UMAP
# n_components = [50, 100, 150, 200, 250]

# # for n in n_components:
# #     umap = UMAP(n_components=n)
# # X_train = umap.fit_transform(X_train_scaled)
# # X_test = umap.transform(X_test_scaled)

# pipe = Pipeline([
# ('ica', FastICA()),
# ('clf', LogisticRegression(max_iter=500, penalty='l1', solver='saga', C=0.01))
# ])

# param_grid = [
#     {
#         'ica__n_components': n_components,
#     },
# ]

# grid = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=5)
# grid.fit(X_train_scaled, y_train)
# # evaluation metric is accuray 
# mean_scores = np.array(grid.cv_results_['mean_test_score'])
# print("With ICA and l1:")
# print(grid.cv_results_['params'])
# print(mean_scores)
# print(grid.best_params_)

#-------------------------------------------------------

## ICA
# n_components = [50, 100, 150, 200, 250]

# for n in n_components:
#     ica = FastICA(n_components=n)
#     X_train = ica.fit_transform(X_train_scaled)
#     X_test = ica.transform(X_test_scaled)

#     param_grid = [
#         {
#             'C': C_options,
#         },
#     ]

#     grid = GridSearchCV(LogisticRegression(max_iter=500, penalty='l2', solver='saga'), param_grid=param_grid, scoring="accuracy", cv=4, n_jobs=-1)
#     grid.fit(X_train, y_train)
#     # evaluation metric is accuray 
#     mean_scores = np.array(grid.cv_results_['mean_test_score'])
#     print("With ICA=",n,"and l2:")
#     print(grid.cv_results_['params'])
#     print(mean_scores)
#     print(grid.best_params_)


# alpha=[0.001,0.01, 0.1]
# for a in alpha:
#     sel_ = SelectFromModel(Lasso(alpha=a))
#     sel_.fit(X_train_scaled, y_train)
#     X_train_selected = sel_.transform(X_train_scaled)
#     X_test_selected = sel_.transform(X_test_scaled)

#     param_grid = [
#         {
#             'C': C_options
#         },
#     ]

#     ####Use L1 LR for clf
#     grid = GridSearchCV(LogisticRegression(max_iter=500, penalty='l2', solver='saga'), param_grid=param_grid, scoring="accuracy", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=3)
#     grid.fit(X_train_selected, y_train)
#     mean_scores = np.array(grid.cv_results_['mean_test_score'])
#     print("FS using Regularisation with alpha=", a, "and l2:")
#     print(grid.cv_results_['params'])
#     print(mean_scores)
#     print(grid.best_params_)


# ###TEST for FS using different
# ####Lasso:
# sel_lasso = SelectFromModel(Lasso(alpha=0.01, tol=0.01))
# sel_lasso.fit(X_train_scaled, y_train)
# X_train_selected = sel_lasso.transform(X_train_scaled)
# X_test_selected = sel_lasso.transform(X_test_scaled)
# clf_lasso = LogisticRegression(max_iter=1000, penalty='l2', solver='saga', C=100).fit(X_train_selected, y_train)
# y_pred_lasso = clf_lasso.predict(X_test_selected)

# ####LogReg:
# sel_logreg = SelectFromModel(LogisticRegression(C=1000, penalty='l1', solver='saga', tol=0.01))
# sel_logreg.fit(X_train_scaled, y_train)
# X_train_selected = sel_logreg.transform(X_train_scaled)
# X_test_selected = sel_logreg.transform(X_test_scaled)
# clf_logreg = LogisticRegression(max_iter=1000, penalty='l2', solver='saga', C=0.1).fit(X_train_selected, y_train)
# y_pred_logreg = clf_logreg.predict(X_test_selected)

# print("Accuracy of LassoFS+LogReg:", accuracy_score(y_test, y_pred_lasso))
# print("Accuracy of LogRegFS+LogReg:", accuracy_score(y_test, y_pred_logreg))

##Doc:
# Accuracy of LassoFS+LogReg: 0.6704545454545454
# Accuracy of LogRegFS+LogReg: 0.625

####### Documentation:
# LR
#Use L1 regulariser:
#Accuracy of C = 0.01: 0.7169811
#Accuracy of C = 0.1: 0.5283018 (does not converge)
#Accuracy of C = 1: 0.0.5471698 (does not converge)
#Accuracy of C = 1.5: 0.5471698 (does not converge)
#Accuracy of C = 10: 0.5471698 (does not converge)
#Accuracy of C = 100: 0.5471698 (does not converge)

#Use L2 regulariser:
#Accuracy of C = 0.01: 0.5471698 (does not converge)
#Accuracy of C = 0.1: 0.5471698 (does not converge)
#Accuracy of C = 1: 0.5471698 (does not converge)
#Accuracy of C = 1.5: 0.5471698 (does not converge)
#Accuracy of C = 10: 0.5471698 (does not converge)
#Accuracy of C = 100: 0.5471698 (does not converge)

#### PCA+LR
#PCA and L1:
# PCA__n_component: 100
# L1__C: 0.1
# Acc = 0.52298
###
#PCA and L2:
# PCA__n_component: 100
# L2__C: 0.01
# Acc = 0.54885057
###
# Best estimator: PCA__n_component=100, C=0.01, penalty=l2


####UMAP+LR
#UMAP and L1:
# Best: n = 50, C = 0.01, acc=0.69254658
#UMAP and L2:
# Best: n = 150, C = 0.01, acc=0.69540373
###
#Best estimator: UMAP__n_component=150, C=0.01, penalty=l2

####ICA+LR ##Try this one again on test set
#ICA and L1:
# Best: n = 100, C = 0.01, acc=0.64367816
#ICA and L2:
# Best: n =50, C = 0.01, acc=0.64942529 
#Best estimator: UMAP__n_component=50, C=0.01, penalty=l2

####Regularisation FS
## L1 reg FS using logistic regression
    ###Regularisation using L1 with diff C values, and then LR as CLF with different C values:
    # Best estimator: Lasso C = 1000, CLF C = 0.1, acc = 0.7644
    ###
    ###Regularisation using L 1with diff C values, and then LR as CLF with different C values:
    # Best estimator: Lasso C = 100, CLF C = 0.1, acc = 0.792966

    
    
    
####USE THIS
## L2 reg FS using Lasso (i.e. linear regression)
    ###Regularisation using L1 with diff alpha values, and then LR as CLF with different C values:
    # Best estimator: Lasso alpha = 0.01, CLF C = 100, acc = 1.0
    ###
    ###Regularisation using L1 with diff alpha values, and then LR as CLF with different C values:
    # Best estimator: Lasso alpha = 0.01, CLF C = 100, acc = 1.0 (this one has two '1.0' prediction)