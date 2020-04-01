## Documentation for hyperparameter tuning and performance on training set
---------- 
### DR + ML methods
*****************
- DR+LR:
*****************
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
>PCA = 3:

Confusion matrix of PPMI testing set:
[[15 11]
 [22 40]]
 {'num_estmtr': 30, 'col_ratio': 0.7, 'subsample_ratio': 0.3, 'max_depth': 3}
 
>PCA = 4:

For n_compo= 4 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 10, 'col_ratio': 0.5, 'subsample_ratio': 0.3, 'max_depth': 3}
[[18  8]
 [28 34]]

>PCA = 5:

Current score: 0.6200941526701634
Confusion matrix of PPMI testing set:
[[16 10]
 [22 40]]
precision of testing set: 0.8
{'num_estmtr': 30, 'col_ratio': 0.7, 'subsample_ratio': 0.3, 'max_depth': 4}

> PCA = 6:

Confusion matrix of PPMI testing set:
[[16 10]
 [25 37]]
precision of testing set: 0.7872340425531915
{'num_estmtr': 200, 'col_ratio': 0.5, 'subsample_ratio': 0.5, 'max_depth': 4}
Current score: 0.6847733697048767

>PCA = 7:

Confusion matrix of PPMI testing set:
[[15 11]
 [24 38]]
precision of testing set: 0.7755102040816326
{'num_estmtr': 150, 'col_ratio': 0.7, 'subsample_ratio': 0.3, 'max_depth': 4}

> PCA = 8:

Confusion matrix of PPMI testing set:
[[16 10]
 [27 35]]
precision of testing set: 0.7777777777777778
{'num_estmtr': 30, 'col_ratio': 0.5, 'subsample_ratio': 0.3, 'max_depth': 3}

> PCA = 9:

Current score: 0.685133431968875
Confusion matrix of PPMI testing set:
[[13 13]
 [16 46]]
precision of testing set: 0.7796610169491526
{'num_estmtr': 200, 'col_ratio': 0.5, 'subsample_ratio': 0.7, 'max_depth': 3}


> PCA = 10:

Confusion matrix of PPMI testing set:
[[14 12]
 [19 43]]
precision of testing set: 0.7818181818181819
{'num_estmtr': 70, 'col_ratio': 0.5, 'subsample_ratio': 0.5, 'max_depth': 3}

For n_compo= 10 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 30, 'col_ratio': 0.7, 'subsample_ratio': 0.3, 'max_depth': 4}
[[16 10]
 [16 46]]


> PCA = 12:

Confusion matrix of PPMI testing set:
[[13 13]
 [14 48]]
precision of testing set: 0.7868852459016393
{'num_estmtr': 150, 'col_ratio': 0.7, 'subsample_ratio': 0.5, 'max_depth': 4}
Current score: 0.6977413419044924

-------------------
> PCA = 13:
Confusion matrix of PPMI testing set:
[[16 10]
 [14 48]]
precision of testing set: 0.8275862068965517
{'num_estmtr': 30, 'col_ratio': 0.7, 'subsample_ratio': 0.3, 'max_depth': 4}

--------------

#### UMAP + XGBoost

----------------------
For UMAP n_compo= 3 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 50, 'col_ratio': 0.7, 'subsample_ratio': 0.3, 'max_depth': 4}
[[21  5]
 [30 32]]
precision of testing set: 0.864864864864

----------------------

For UMAP n_compo= 6 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 30, 'col_ratio': 0.5, 'subsample_ratio': 0.7, 'max_depth': 2}
[[19  7]
 [26 36]]
precision of testing set: 0.8372093023
 
For UMAP n_compo= 9 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 50, 'col_ratio': 0.1, 'subsample_ratio': 0.7, 'max_depth': 2}
[[18  8]
 [25 37]]
precision of testing set: 0.82222222222
 
For UMAP n_compo= 10 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 30, 'col_ratio': 0.1, 'subsample_ratio': 0.7, 'max_depth': 2}
[[19  7]
[28 34]]
precision: 0.8292682926
 
For UMAP n_compo= 11 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 70, 'col_ratio': 0.7, 'subsample_ratio': 0.5, 'max_depth': 4}
[[17  9]
 [18 44]]
precision: 0.83018868
 
For UMAP n_compo= 12 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 10, 'col_ratio': 0.5, 'subsample_ratio': 0.7, 'max_depth': 4}
[[20  6]
 [24 38]]
precision of testing set: 0.8636363636363636

----------------------
>For UMAP n_compo= 13 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 50, 'col_ratio': 0.7, 'subsample_ratio': 0.3, 'max_depth': 4}
[[21  5]
 [27 35]]
 
 ----------------------

#### ICA+ XGBoost
For ICA n_compo= 3 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 10, 'col_ratio': 0.1, 'subsample_ratio': 0.5, 'max_depth': 3}
Confusion matrix of PPMI training set:
[[167  80]
 [ 78 169]]
Confusion matrix of PPMI testing set:
[[14 12]
 [22 40]]
precision of testing set: 0.7692307692307693

--------------------------------
>For ICA n_compo= 5 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 10, 'col_ratio': 0.1, 'subsample_ratio': 0.5, 'max_depth': 3}
 Confusion matrix of PPMI training set:
[[165  82]
 [ 66 181]]
Confusion matrix of PPMI testing set:
[[18  8]
 [28 34]]
precision of testing set: 0.8095238095238095

--------------------------------
precision of testing set: 0.7755102040816326
For ICA n_compo= 9 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 300, 'col_ratio': 0.5, 'subsample_ratio': 0.3, 'max_depth': 4}
Confusion matrix of PPMI training set:
[[247   0]
 [  0 247]]
[[16 10]
 [26 36]]
precision of testing set: 0.782608695652174

For ICA n_compo= 11 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 100, 'col_ratio': 0.3, 'subsample_ratio': 0.5, 'max_depth': 2}
Confusion matrix of PPMI training set:
[[225  22]
 [ 28 219]]
Confusion matrix of PPMI testing set:
[[15 11]
 [24 38]]
precision of testing set: 0.7755102040816326
 
 
For ICA n_compo= 13 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 10, 'col_ratio': 0.3, 'subsample_ratio': 0.3, 'max_depth': 3}
Confusion matrix of PPMI training set:
[[188  59]
 [ 62 185]]
Confusion matrix of PPMI testing set:
[[15 11]
 [26 36]]
precision of testing set: 0.7659574468085106


#### FS + XGBoost

>For FS alpha= 0.005 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 10, 'col_ratio': 0.3, 'subsample_ratio': 0.7, 'max_depth': 3}
Confusion matrix of PPMI training set:
[[240   7]
 [  3 244]]
Confusion matrix of PPMI testing set:
[[14 12]
 [15 47]]
precision of testing set: 0.7966101694915254


For FS alpha= 0.01 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 30, 'col_ratio': 0.3, 'subsample_ratio': 0.3, 'max_depth': 4}
Confusion matrix of PPMI training set:
[[245   2]
 [  0 247]]
Confusion matrix of PPMI testing set:
[[13 13]
 [ 9 53]]
precision of testing set: 0.803030303030303


For FS alpha= 0.04 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 70, 'col_ratio': 0.3, 'subsample_ratio': 0.5, 'max_depth': 2}
Confusion matrix of PPMI training set:
[[247   0]
 [  0 247]]
Confusion matrix of PPMI testing set:
[[11 15]
 [ 6 56]]
precision of testing set: 0.7887323943661971


#### Clustering + FS + XGBoost
cluster: 6
FS: a = 0.005, tol = 0.01, random_seed = 42
XGB: n_estmt=50, learning_rate=0.3, colsample=0.5, subsample=0.5, max_depth=3
Confusion matrix of clustered + reduced training set using UMAP+XGBoost:
[[242   5]
 [  0 247]]
Confusion matrix of clustered + reduced testing set using UMAP+XGBoost:
[[14 12]
 [ 1 61]]
precision of testing set: 0.8356164383561644

FS: a = 0.005, tol = 0.01, random_seed = 42
XGB: n_estmt=70, learning_rate=0.5, colsample=0.4, subsample=0.3, max_depth=3
Confusion matrix of clustered + reduced training set using FS+XGBoost:
[[239   8]
 [  0 247]]
Confusion matrix of clustered + reduced testing set using FS+XGBoost:
[[17  9]
 [ 4 58]]
precision of testing set: 0.8656716417910447

#### Clustering + FS + LR
cluster: 6
FS: a = 0.005, tol = 0.01, random_seed = 42
LR: C = 1, penalty = l1
Confusion matrix of clustered + reduced training set using FS+LR:
[[247   0]
 [  0 247]]
Confusion matrix of clustered + reduced testing set using FS+LR:
[[26  0]
 [13 49]]
precision of testing set: 1.0

FS: a = 0.005, tol = 0.01, random_seed = 42
LR: C = 1, penalty = l2
Confusion matrix of clustered + reduced training set using FS+LR:
[[247   0]
 [  0 247]]
Confusion matrix of clustered + reduced testing set using FS+LR:
[[26  0]
 [ 2 60]]
precision of testing set: 1.0

#### Clustering + FS + SVM
cluster: 6
FS: a = 0.08, tol = 0.01, random_seed = 42
SVM: C = 0.0001, kernel = poly, gamma=1.5
Confusion matrix of clustered + reduced training set using FS+LR:
[[237  10]
 [  0 247]]
Confusion matrix of clustered + reduced testing set using FS+LR:
[[13 13]
 [ 1 61]]
precision of testing set: 0.8243243243243243

#### VAE