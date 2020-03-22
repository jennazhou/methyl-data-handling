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