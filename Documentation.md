## Documentation for hyperparameter tuning and performance on training set
---------- 
### DR + ML methods
*****************
- DR+LR:
*****************
------------
#### Accuracy as metrics (which should not be used)
------------
#### PCA+LR
* PCA and L1:
PCA__n_component: 100 
L1__C: 0.1
Acc = 0.52298

* PCA and L2:
PCA__n_component: 100
L2__C: 0.01
Acc = 0.54885057

> Best estimator: PCA__n_component=100, C=0.01, penalty=l2


#### UMAP+LR
* UMAP and L1:
Best: n = 50, C = 0.01, acc=0.69254658
* UMAP and L2:
Best: n = 150, C = 0.01, acc=0.69540373

> Best estimator: UMAP__n_component=150, C=0.01, penalty=l2

#### ICA+LR 
* ICA and L1:
Best: n = 100, C = 0.01, acc=0.64367816
* ICA and L2:
Best: n =50, C = 0.01, acc=0.64942529 

> Best estimator: UMAP__n_component=50, C=0.01, penalty=l2

#### Regularisation FS
* L2 reg FS using Lasso (i.e. linear regression)
Regularisation using L1 with diff alpha values, and then LR as CLF with different C values:
Best estimator: Lasso alpha = 0.01, CLF C = 100, acc = 1.0

> Regularisation using L1 with diff alpha values, and then LR as CLF with different C values:
  Best estimator: Lasso alpha = 0.01, CLF C = 100, acc = 1.0 (this one has two '1.0' prediction)
    
------------
### F1 score as metrics (the higher the more desirable)
#### with upsampling of the dataset
------------

------------
### Precision score as metrics (the higher the more desirable)
#### with upsampling of the dataset
------------

#### PCA+LR (StandardScaler)

Total variance covered: 0.6462956834157272
Mean score of precision of the best C: 0.626496369787509
With PCA= 40 and l2, the best params are:
{'C': 0.1/0.01} for n_compo= 40
[[153  63]
 [ 58 158]]
 
[[15 23]
 [33 60]]
 
precision of testing set: 0.7228915662650602
recall of testing set: 0.6451612903225806
accuracy of testing set: 0.7199074074074074

#### UMAP + LR
With UMAP= 60 and l1, the best params are:
{'C': 100} for n_compo= 60
Confusion matrix of PPMI training set:
[[132 115]
 [ 76 171]]
Confusion matrix of PPMI testing set:
[[17  9]
 [24 38]]

With UMAP = 40 and l2, the best params are:
{'C': 100} for n_compo= 40
Confusion matrix of PPMI training set:
[[150  97]
 [ 85 162]]
Confusion matrix of PPMI testing set:
[[14 12]
 [30 32]]

#### ICA + LR
With ICA= 40 and l1, the best params are:
{'C': 100} for n_compo= 40
Confusion matrix of PPMI training set:
[[187  60]
 [ 65 182]]
Confusion matrix of PPMI testing set:
[[14 12]
 [19 43]]

With ICA= 40 and l2, the best params are:
{'C': 1000} for n_compo= 40
Confusion matrix of PPMI training set:
[[185  62]
 [ 66 181]]
Confusion matrix of PPMI testing set:
[[14 12]
 [20 42]]


#### Lasso FS + LR
With Lasso(L2) FS alpha= 0.09 and l1, the best params are:
{'C': 1} for alpha= 0.09
Confusion matrix of PPMI training set:
[[224  23]
 [ 16 231]]
Confusion matrix of PPMI testing set:
[[13 13]
 [13 49]]
 
With Lasso(L2) FS alpha= 0.08 and l2, the best params are:
{'C': 1} for alpha= 0.08
Confusion matrix of PPMI training set:
[[240   7]
 [  3 244]]
Confusion matrix of PPMI testing set:
[[11 15]
 [13 49]] 
 
 
 
 
#### PCA + SVM
SVM with PCs= 16 kernel poly and gamma 1.5 has best performance of 0.760747071858182
9 with {'C': 0.001}

Current clf: SVC(C=1.0, break_ties=False, cache_size=200, class_weight='balanced', c
oef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='poly',
    max_iter=3000, probability=False, random_state=None, shrinking=True,
    tol=0.01, verbose=False)
Confusion matrix of PPMI training set:
[[246   1]
 [ 12 235]]
Confusion matrix of PPMI testing set:
[[10 16]
 [13 49]]
 
 #### UMAP + SVM

SVM with UMAPs= 18 kernel rbf and gamma 1.5 has best performance of 0.6435273033944761 with {'C': 1}
Current clf: SVC(C=1.0, break_ties=False, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape='ovr', degree=3, gamma=1.5, kernel='rbf',max_iter=3000, probability=False, random_state=None, shrinking=True, tol=0.01, verbose=False)

Confusion matrix of PPMI training set:
[173  74]
[ 71 176]]
Confusion matrix of PPMI testing set:
[[14 12]
[30 32]]


#### ICA + SVM

SVM with ICAs= 16 kernel rbf and gamma 1.5 has best performance of 0.6910791302964814 with {'C': 1000}

Current clf: SVC(C=1.0, break_ties=False, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf',
    max_iter=3000, probability=False, random_state=None, shrinking=True,
    tol=0.01, verbose=False)
Confusion matrix of PPMI training set:
[[130 117]
 [ 74 173]]
Confusion matrix of PPMI testing set:
[[10 16]
 [22 40]]
 
 
#### FS + SVM
SVM with Lasso FS alpha= 0.1 kernel poly and gamma 1.5 has best performance of 0.953568479324688 with {'C': 0.001}

Current clf: SVC(C=1.0, break_ties=False, cache_size=200, class_weight='balanced', c
oef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1.5, kernel='poly',
    max_iter=3000, probability=False, random_state=None, shrinking=True,
    tol=0.01, verbose=False)
Confusion matrix of PPMI training set:
[[247   0]
 [  0 247]]
Confusion matrix of PPMI testing set:
[[14 12]
 [26 36]]


#### PCA + XGBoost
XGBoost with n_compo= 10 , num_estmtr= 100 ,col_ratio= 0.5 has best performance of 0.7469201364518785 with {'max_depth': 5}
Current score: 0.7469201364518785
For 10 compo PCA, cur params are:
{'num_estmtr': 100, 'col_ratio': 0.5, 'learning_rate': 0.3, 'max_depth': 5}
Confusion matrix of PPMI training set:
[[247   0]
 [  0 247]]
Confusion matrix of PPMI testing set:
[[11 15]
 [11 51]]


XGBoost with n_compo= 20 , num_estmtr= 50 ,col_ratio= 0.7 has best performance of 0.8020109964600438 with {'max_depth': 5}
Current score: 0.8020109964600438
For 20 compo PCA, cur params are:
{'num_estmtr': 50, 'col_ratio': 0.7, 'learning_rate': 0.3, 'max_depth': 5}
Confusion matrix of PPMI training set:
[[247   0]
 [  0 247]]
Confusion matrix of PPMI testing set:
[[ 8 18]
 [ 2 60]]