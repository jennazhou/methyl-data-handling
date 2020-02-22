import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.decomposition import FastICA

from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

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
    
# Train XGBoost classifier
# DMatrix: a data structure that makes everything more efficient
# dtrain = xgb.DMatrix(X_train_scaled, y_train)
# dtest = xgb.DMatrix(X_test_scaled, y_test)


# print('Already skimmed parameters')
# num_round = 50
# bst = xgb.train(param, dtrain, num_round, [(dtest, 'test'), (dtrain, 'train')])
# # y_pred = bst.predict(dtest)
# bst

### PCA
# n_components = [50, 100, 150, 200, 250]
# for n in n_components:
#     pca = PCA(n_components=n)
#     X_train = pca.fit_transform(X_train_scaled)
#     X_test = pca.transform(X_test_scaled)



# ## UMAP
# n_components = [250]
# for n in n_components:
#     umap = UMAP(n_components=n)
#     X_train = umap.fit_transform(X_train_scaled)
#     X_test = umap.transform(X_test_scaled)

## iCA
# n_components = [50, 100, 150, 200, 250]
# for n in n_components:
#     ica = FastICA(n_components=n)
#     X_train = ica.fit_transform(X_train_scaled)
#     X_test = ica.transform(X_test_scaled)

# FS using regularisation
# lr_c = [1]
# for C in lr_c:
#     sel_ = SelectFromModel(LogisticRegression(C=C, penalty='l1', solver='saga', tol=0.1))
#     sel_.fit(X_train_scaled, y_train)
#     X_train = sel_.transform(X_train_scaled)
#     X_test = sel_.transform(X_test_scaled)
#     print(X_train.shape)

alpha=[0.0001]
for a in alpha:
    sel_ = SelectFromModel(Lasso(alpha=a, tol=0.01, random_state=42))
    sel_.fit(X_train_scaled, y_train)
    X_train = sel_.transform(X_train_scaled)
#     X_test_selected = sel_.transform(X_test_scaled)
    print("Shape of training set with alpha=", a, ":", X_train.shape)
    
    
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
    
#     dtrain = xgb.DMatrix(X_train, y_train)
#     dtest = xgb.DMatrix(X_test, y_test)
#     param = {
#         'objective' : 'binary:logistic', 
#         'seed' : 42, 
#         'tree_method':'gpu_hist'
#         }

 
    colsample_bytree = [ 0.3, 0.5 , 0.7 ]
    ne = [70, 100, 150, 200, 300] 
    print("Number of values for estimators:", len(ne))
    #A np array to store all perf to find the best one
    best_perf=0
    #A list to store the dictionary of the best estimator
    
    for n_estimator in ne:
        for colsample in colsample_bytree:
            xgb_clf = xgb.XGBClassifier(
                objective='binary:logistic', 
                seed=42, 
#                 tree_method='gpu_hist',
                learning_rate=0.3,
#                 gpu_id=2,
                colsample_bytree=colsample,
                n_estimators=n_estimator
            )

            param_grid = {
                "max_depth": [3],#, 5, 8
    #             "colsample_bytree": [ 0.3, 0.5 , 0.7 ],
    #             "n_estimators":[5, 8, 10] # 30, 40, 50, 70
               }

            grid = GridSearchCV(xgb_clf, param_grid=param_grid, scoring="accuracy", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=3)
            grid.fit(X_train, y_train)
            print('XGBoost with FS reg alpha value=', a, ' n_trees=', n_estimator, 'and colsample=', colsample, ':')
            print('Highest accuracy:',grid.best_score_)
            print('Best estimator: ', grid.best_params_)
            print()
            if grid.best_score_ > best_perf:
                best_perf = grid.best_score_
                best_param = grid.best_params_
                tree_flag = n_estimator
                col_ratio_flag = colsample

    print("XGBoost with", a, 'alpha value for Reg FS has best performance of',best_perf, "with", best_param, tree_flag, "trees and ", col_ratio_flag, "colsample ratio.")
    print()


##############Documentation
#####XGBoost on original data
# XGBoost with 0.3 colsample_bytree :
# Highest accuracy: 0.6925287356
# Best estimator:  {'max_depth': 3, 'n_estimator': 10}

# XGBoost with 0.5 colsample_bytree :
# Highest accuracy: 0.69827586206
# Best estimator:  {'max_depth': 5, 'n_estimator': 10}

# XGBoost with 0.7 colsample_bytree :
# Highest accuracy: 0.6937235636
# Best estimator:  {'max_depth': 3, 'n_estimator': 10}

#####XGBoost with DR techniques
###Common trend: as the number of tree increase, performance increases. After reaching peak, as number of trees increases, performance decreases probably due to overfitting

#PCA
# XGBoost with 50 PCs has best performance of 0.7068965517241379 with {'colsample_bytree': 0.3, 'max_depth': 5} and 30 trees.
# XGBoost with 100 PCs has best performance of 0.7011494252873564 with {'colsample_bytree': 0.3, 'max_depth': 3} and 10 trees.
# XGBoost with 150 PCs has best performance of 0.7183908045977012 with {'colsample_bytree': 0.5, 'max_depth': 3} and 50 trees.
# XGBoost with 200 PCs has best performance of 0.7183908045977012 with {'colsample_bytree': 0.5, 'max_depth': 8} and 70 trees.
# XGBoost with 250 PCs has best performance of 0.7097701149425287 with {'colsample_bytree': 0.3, 'max_depth': 8} and 30 trees.

#UMAP
# XGBoost with 50 UMAP clusters has best performance of 0.6982758620689654 with {'colsample_bytree': 0.7, 'max_depth': 3} and 20 trees.
# XGBoost with 100 UMAP clusters has best performance of 0.6982758620689656 with {'colsample_bytree': 0.5, 'max_depth': 3} and 5 trees.
# XGBoost with 150 UMAP clusters has best performance of 0.6896551724137931 with {'colsample_bytree': 0.7, 'max_depth': 5} and 10 trees.
# XGBoost with 200 UMAP clusters has best performance of 0.7097701149425287 with {'colsample_bytree': 0.7, 'max_depth': 3} and 30 trees.
# XGBoost with 250 UMAP clusters has best performance of 0.6896551724137931 with {'colsample_bytree': 0.3, 'max_depth': 3} and 8 trees.

#####ICA
# XGBoost with 50 ICs has best performance of 0.7068965517241379 with {'colsample_bytree': 0.5, 'max_depth': 8} and 50 trees.
# XGBoost with 100 ICs has best performance of 0.7155172413793104 with {'colsample_bytree': 0.3, 'max_depth': 5} and 20 trees.
# XGBoost with 150 ICs has best performance of 0.6925287356321839 with {'colsample_bytree': 0.3, 'max_depth': 5} and 30 trees.
# XGBoost with 200 ICs has best performance of 0.7011494252873564 with {'colsample_bytree': 0.3, 'max_depth': 5} and 30 trees.
# XGBoost with 250 ICs has best performance of 0.7126436781609197 with {'colsample_bytree': 0.3, 'max_depth': 8} and 5(100) trees.



#XGBoost on selected features using regularisation
##Note: computation cost increases significantly without much increase in the accuracy level
# XGBoost with 0.1 C value for Reg FS has best performance of 0.7212643678160919 with {'colsample_bytree': 0.3, 'max_depth': 5} and 100 trees.
#Num_of_features: 17454

# XGBoost with 1 C value for Reg FS has best performance of 0.7068965517241379 with {'colsample_bytree': 0.7, 'max_depth': 3} and 30 trees. 
#Num_of_features: 153327

# XGBoost with 10 C value for Reg FS has best performance of 0.7126436781609197 with 
# {'colsample_bytree': 0.5, 'max_depth': 3} and 70 trees.
#Num_of_features: 185805

# XGBoost with 100 C value for Reg FS has best performance of 0.7183908045977011 with {'colsample_bytree': 0.7, 'max_depth': 5} and 50 trees.
#Num_of_features: 163107

# XGBoost with 1000 C value for Reg FS has best performance of 0.7011494252873564 with {'colsample_bytree': 0.7, 'max_depth': 3} and 20 trees.
#Num_of_features: 189971

########Use lasso library
# XGBoost on selected features using Reg with Lasso
# Shape of training set with alpha= 0.1 : (348, 4)
# XGBoost with 0.1 alpha value for Reg FS has best performance of 0.6925287356321839 with {'max_depth': 3} 70 trees and  0.5 colsample ratio.

# Shape of training set with alpha= 0.01 : (348, 323)
# XGBoost with 0.01 alpha value for Reg FS has best performance of 0.7988505747126436 with {'max_depth': 3} 100 trees and  0.3 colsample ratio.

# Shape of training set with alpha= 0.001 : (348, 809)
# XGBoost with 0.001 alpha value for Reg FS has best performance of 0.7729885057471265 with {'max_depth': 3} 100 trees and  0.3 colsample ratio.


#####Time#####
#In XGBoost, the increase in time is very significant and positively correlated with increase in the number of features used for classification