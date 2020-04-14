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

-------------
### Variance covered by different PCs
-------------
30:
Total variance of dataset covered: 0.6078676049570196
40:
Total variance of dataset covered: 0.6374173508172822
50:
Total variance of dataset covered: 0.6614219167255786
60:
Total variance of dataset covered: 0.6836696320603294
70:
Total variance of dataset covered: 0.7036559740342414
80:
Total variance of dataset covered: 0.7220859447204653
100:
Total variance of dataset covered: 0.7552134425631912
120:
Total variance of dataset covered: 0.7838099132523845
150:
Total variance of dataset covered: 0.8212265973924266
200:
Total variance of dataset covered: 0.8755816915332136
250:
Total variance of dataset covered: 0.9234137877212798
280:
Total variance of dataset covered: 0.9494832410074734
300:
Total variance of dataset covered: 0.9659308941841616
320:
Total variance of dataset covered: 0.9814173586754761
350:
Total variance of dataset covered: 0.9999999999999992


Results for PCA compo with variance > 87% (i.e. n = 200, 250, 280)
n = 200 + LR:
Mean score of precision of the best C: 0.8609848484848485
Confusion matrix of PPMI testing set:
[[ 2 24]
 [ 3 59]]
precision of testing set: 0.7108433734939759
{'n_component': 200}
{'C': 0.001}

Mean score of precision of the best C: 0.86431989063568
Confusion matrix of PPMI testing set:
[[ 1 25]
 [ 2 60]]
precision of testing set: 0.7058823529411765
{'n_component': 250}
{'C': 0.001}

Mean score of precision of the best C: 0.8700319655375836
Confusion matrix of PPMI testing set:
[[ 1 25]
 [ 2 60]]
precision of testing set: 0.7058823529411765
{'n_component': 280}
{'C': 0.001}

------------
### Precision score as metrics (the higher the more desirable)
#### with upsampling of the dataset
------------

#### PCA+LR (StandardScaler)
{'n_component': 30, 'C': 0.01, 'penalty': 'l2'}
Confusion matrix of PPMI training set:
[[175  72]
 [ 60 187]]
Confusion matrix of PPMI testing set:
[[11 15]
 [21 41]]
 
{'n_component': 30, 'C': 1, 'penalty': 'l1'}
Confusion matrix of PPMI training set:
[[175  72]
 [ 60 187]]
Confusion matrix of PPMI testing set:
[[11 15]
 [21 41]]
 

{'n_component': 30, 'C': 0.01, 'penalty': 'l2'}

{'n_component': 50, 'C': 0.01, 'penalty': 'l2'}
Confusion matrix of PPMI training set:
[[186  61]
 [ 62 185]]
Confusion matrix of PPMI testing set:
[[11 15]
 [20 42]]
 
{'n_component': 50, 'C': 1, 'penalty': 'l1'}
Confusion matrix of PPMI training set:
[[186  61]
 [ 62 185]]
Confusion matrix of PPMI testing set:
[[11 15]
 [20 42]]
 
#### ICA + LR
{'n_component': 40, 'C': 100, 'penalty': 'l1'}
Confusion matrix of PPMI training set:
[[187  60]
 [ 65 182]]
Confusion matrix of PPMI testing set:
[[14 12]
 [19 43]]

{'n_component': 40, 'C': 1000, 'penalty': 'l2'}
Confusion matrix of PPMI training set:
[[185  62]
 [ 66 181]]
Confusion matrix of PPMI testing set:
[[14 12]
 [20 42]]

{'n_component': 30, 'C': 100, 'penalty': 'l1'}
Confusion matrix of PPMI training set:
[[164  83]
 [ 64 183]]
Confusion matrix of PPMI testing set:
[[13 13]
 [22 40]]
 
{'n_component': 40, 'C': 1000, 'penalty': 'l2'}
{'n_component': 40, 'C': 100, 'penalty': 'l2'}

 

#### UMAP + LR
{'n_neighbour': 20, 'min_dist': 0.4, 'n_component': 40, 'C': 1000, 'penalty': 'l1'}
Confusion matrix of PPMI training set:
[[148  99]
 [ 82 165]]
Confusion matrix of PPMI testing set:
[[19  7]
 [32 30]]
 
 {'n_neighbour': 20, 'min_dist': 0.25, 'n_component': 40, 'C': 100, 'penalty': 'l1'}
Confusion matrix of PPMI training set:
[[153  94]
 [ 84 163]]
Confusion matrix of PPMI testing set:
[[17  9]
 [31 31]]
 
 For UMAP n_compo=50,from confusion matrix of PPMI testing set, best params are: 
{'n_neighbour': 20, 'min_dist': 0.5, 'n_component': 50, 'C': 1000, 'penalty': 'l2'}
[[14 12]
 [30 32]]
 
{'n_neighbour': 15, 'min_dist': 0.4, 'n_component': 60, 'C': 100, 'penalty': 'l1'}
Confusion matrix of PPMI training set:
[[156  91]
 [ 84 163]]
Confusion matrix of PPMI testing set:
[[12 14]
 [25 37]]
precision of testing set:0.7254901960784313
The temp confmatx of testing set has been updated to:
[[12 14]
 [25 37]]

#### Lasso FS + LR
{'lasso_a': 0.09, 'C': 1, 'penalty': 'l1'}
Confusion matrix of PPMI training set:
[[224  23]
 [ 16 231]]
Confusion matrix of PPMI testing set:
[[13 13]
 [13 49]]
precision of testing set:0.7903225806451613
The temp confmatx of testing set has been updated to:
[[13 13]
 [13 49]]
 
{'lasso_a': 0.08, 'C': 1, 'penalty': 'l2'}
Confusion matrix of PPMI training set:
[[240   7]
 [  3 244]]
Confusion matrix of PPMI testing set:
[[11 15]
 [13 49]]

{'lasso_a': 0.08, 'C': 1, 'penalty': 'l1'}
Confusion matrix of PPMI training set:
[[240   7]
 [  5 242]]
Confusion matrix of PPMI testing set:
[[11 15]
 [14 48]]
 
 
 
 
 
 
 
 
 
#### PCA + SVM
 
 {'n_component': 10, 'kernel': 'poly', 'gamma': 0.0001, 'coef0': 10, 'C': 0.001}
Confusion matrix of PPMI training set:
[[221  26]
 [ 32 215]]
Confusion matrix of PPMI testing set:
[[13 13]
 [13 49]]
 
{'n_component': 14, 'kernel': 'poly', 'gamma': 0.0001, 'coef0': 0.5, 'C': 0.01}
Confusion matrix of PPMI training set:
[[243   4]
 [ 15 232]]
Confusion matrix of PPMI testing set:
[[13 13]
 [14 48]]
 
{'n_component': 14, 'kernel': 'poly', 'gamma': 0.0001, 'coef0': 10, 'C': 0.01}
Confusion matrix of PPMI training set:
[[245   2]
 [ 16 231]]
Confusion matrix of PPMI testing set:
[[12 14]
 [11 51]]
 
{'n_component': 16, 'kernel': 'poly', 'gamma': 0.001, 'coef0': 3, 'C': 0.001}
Confusion matrix of PPMI training set:
[[247   0]
 [  9 238]]
Confusion matrix of PPMI testing set:
[[11 15]
 [ 9 53]]
 
{'n_component': 16, 'kernel': 'poly', 'gamma': 0.001, 'coef0': 0.5, 'C': 0.001}
Confusion matrix of PPMI training set:
[[244   3]
 [ 12 235]]
Confusion matrix of PPMI testing set:
[[10 16]
 [ 5 57]]
precision of testing set:0.7808219178082192
[[10 16]
 [ 5 57]]
 
 
 
 #### ICA + SVM

{'n_component': 12, 'kernel': 'rbf', 'gamma': 0.01, 'coef0': 0.5, 'C': 1000}
Confusion matrix of PPMI training set:
[[136 111]
 [ 73 174]]
Confusion matrix of PPMI testing set:
[[12 14]
 [19 43]]
 
{'n_component': 12, 'kernel': 'poly', 'gamma': 0.001, 'coef0': 7, 'C': 100}
Confusion matrix of PPMI training set:
[[132 115]
 [ 69 178]]
Confusion matrix of PPMI testing set:
[[12 14]
 [19 43]]
 
{'n_component': 12, 'kernel': 'poly', 'gamma': 0.01, 'coef0': 5, 'C': 1000}
Confusion matrix of PPMI training set:
[[138 109]
 [ 81 166]]
Confusion matrix of PPMI testing set:
[[13 13]
 [19 43]]
 
{'n_component': 12, 'kernel': 'poly', 'gamma': 0.01, 'coef0': 10, 'C': 100}
Confusion matrix of PPMI training set:
[[141 106]
 [ 79 168]]
Confusion matrix of PPMI testing set:
[[13 13]
 [19 43]]
 
{'n_component': 12, 'kernel': 'poly', 'gamma': 1.5, 'coef0': 1, 'C': 1000}
Confusion matrix of PPMI training set:
[[173  74]
 [ 67 180]]
Confusion matrix of PPMI testing set:
[[13 13]
 [17 45]]
 
{'n_component': 12, 'kernel': 'poly', 'gamma': 3, 'coef0': 0.5, 'C': 1000}
Confusion matrix of PPMI training set:
[[203  44]
 [ 40 207]]
Confusion matrix of PPMI testing set:
[[13 13]
 [17 45]]
 
 
 #### UMAP + SVM

{'n_neighbour': 5, 'min_dist': 0.25, 'n_component': 10, 'kernel': 'rbf', 'gamma': 0.01, 'coef0': 0.5, 'C': 1000}
Confusion matrix of PPMI training set:
[[154  93]
 [ 63 184]]
Confusion matrix of PPMI testing set:
[[14 12]
 [16 46]]
 
 {'n_neighbour': 10, 'min_dist': 0.25, 'n_component': 10, 'kernel': 'rbf', 'gamma': 0.01, 'coef0': 7, 'C': 1000}
Confusion matrix of PPMI training set:
[[151  96]
 [ 71 176]]
Confusion matrix of PPMI testing set:
[[17  9]
 [29 33]]
 
{'n_neighbour': 10, 'min_dist': 0.25, 'n_component': 10, 'kernel': 'poly', 'gamma': 0.01, 'coef0': 1, 'C': 1000}
Confusion matrix of PPMI training set:
[[142 105]
 [ 62 185]]
Confusion matrix of PPMI testing set:
[[17  9]
 [27 35]]
 
 {'n_neighbour': 10, 'min_dist': 0.7, 'n_component': 10, 'kernel': 'rbf', 'gamma': 3, 'coef0': 7, 'C': 100}
Confusion matrix of PPMI training set:
[[247   0]
 [  0 247]]
Confusion matrix of PPMI testing set:
[[12 14]
 [ 8 54]]
 
 (can be all values for coef0 for above models)
 
 
#### FS + SVM]

{'lasso_a': 0.08, 'kernel': 'poly', 'gamma': 0.01, 'coef0': 10, 'C': 0.01}
Confusion matrix of PPMI training set:
[[237  10]
 [  3 244]]
Confusion matrix of PPMI testing set:
[[11 15]
 [12 50]]
 
{'lasso_a': 0.1, 'kernel': 'poly', 'gamma': 1, 'coef0': 0.5, 'C': 0.01}
Confusion matrix of PPMI training set:
[[247   0]
 [  0 247]]
Confusion matrix of PPMI testing set:
[[13 13]
 [23 39]]
 
SVM with Lasso FS alpha= 0.1 kernel poly and gamma 1.5, coef0 = 0, has best performance of 0.953568479324688 with {'C': 0.001}
Confusion matrix of PPMI training set:
[[247   0]
 [  0 247]]
Confusion matrix of PPMI testing set:
[[14 12]
 [26 36]]
 
#### PCA + XGBoost

For PCA n_compo=12,from confusion matrix of PPMI testing set, best params are: 
{'n_components': 12, 'num_estmtr': 200, 'col_ratio': 0.8, 'subsample_ratio': 0.3, 'max_depth': 3}
Confusion matrix of PPMI training set:
[[245   2]
 [  1 246]]

[[16 10]
 [16 46]]
 
>{'n_components': 13, 'num_estmtr': 35, 'col_ratio': 0.5, 'subsample_ratio': 0.4, '
max_depth': 4}
[[14 12]
 [15 47]]

>{'n_components': 12, 'num_estmtr': 100, 'col_ratio': 0.7, 'subsample_ratio': 0.4, 'max_dep
th': 4}
[[15 11]
 [16 46]]

>{'n_components': 10,'num_estmtr': 30, 'col_ratio': 0.7, 'subsample_ratio': 0.3, 'max_depth': 4}
[[16 10]
 [16 46]]
 
>{'n_components':13, 'num_estmtr': 30, 'col_ratio': 0.7, 'subsample_ratio': 0.3, 'max_depth': 4}
Confusion matrix of PPMI testing set:
[[16 10]
 [14 48]]
precision of testing set: 0.8275862068965517


#### ICA+ XGBoost
For ICA n_compo= 5 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 10, 'col_ratio': 0.1, 'subsample_ratio': 0.5, 'max_depth': 3}
 Confusion matrix of PPMI training set:
[[165  82]
 [ 66 181]]
Confusion matrix of PPMI testing set:
[[18  8]
 [28 34]]
precision of testing set: 0.8095238095238095


{'n_components': 14, 'num_estmtr': 10, 'col_ratio': 0.4, 'subsample_ratio': 0.4, 'max_depth': 3}
[[15 11]
 [13 49]]
 
 {'n_components': 14, 'num_estmtr': 20, 'col_ratio': 0.7, 'subsample_ratio': 0.3, 'max_depth': 4}
[[17  9]
 [21 41]]
 
 
 
--------------

#### UMAP + XGBoost

 
For UMAP n_compo= 10 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 30, 'col_ratio': 0.1, 'subsample_ratio': 0.7, 'max_depth': 2}
[[19  7]
[28 34]]
precision: 0.8292682926
 
>For UMAP n_compo= 11 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 70, 'col_ratio': 0.7, 'subsample_ratio': 0.5, 'max_depth': 4}
[[17  9]
 [18 44]]
precision: 0.83018868


>For UMAP n_compo= 12 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 10, 'col_ratio': 0.5, 'subsample_ratio': 0.7, 'max_depth': 4}
[[20  6]
 [24 38]]
precision of testing set: 0.8636363636363636
 
 
{'n_neighbours': 3, 'min_dist': 0.4, 'n_components': 10, 'num_estmtr': 20, 'col_ratio': 0.8, 'subsample_ratio': 0.5, 'max_depth': 4}
[[16 10]
 [17 45]]
 
 
---TBC---



#### FS + XGBoost
{'lasso_a': 0.001, 'num_estmtr': 30, 'col_ratio': 0.8, 'subsample_ratio': 0.3, 'max_depth': 2}
[[13 13]
 [11 51]]
 
>For FS alpha= 0.005 ,from confusion matrix of PPMI testing set, best params are:
{'num_estmtr': 10, 'col_ratio': 0.3, 'subsample_ratio': 0.7, 'max_depth': 3}
Confusion matrix of PPMI training set:
[[240   7]
 [  3 244]]
Confusion matrix of PPMI testing set:
[[14 12]
 [15 47]]
precision of testing set: 0.7966101694915254

{'lasso_a': 0.01, 'num_estmtr': 30, 'col_ratio': 0.3, 'subsample_ratio': 0.3, 'max_depth': 4}
[[13 13]
 [ 9 53]]
precision of testing set: 0.803030303030303

{'lasso_a': 0.08, 'num_estmtr': 10, 'col_ratio': 0.7, 'subsample_ratio': 0.4, 'max_depth': 2}
[[14 12]
 [12 50]]


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