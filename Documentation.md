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
{'C': 0.1} for n_compo= 40
[[153  63]
 [ 58 158]]
 
[[15 23]
 [33 60]]
 
precision of testing set: 0.7228915662650602
recall of testing set: 0.6451612903225806
accuracy of testing set: 0.7199074074074074

#### UMAP + LR
With UMAP= 45 and l1, the best params are:
{'C': 1000} for n_compo= 45

[[110 106]
 [ 56 160]]
[[17 21]
 [45 48]]
precision of testing set: 0.6956521739130435
recall of testing set 0.5161290322580645
accuracy of testing set 0.625

With UMAP = 40 and l2, the best params are:
{'C': 0.001} for n_compo= 40
[[121  95]
 [ 87 129]]
[[19 19]
 [30 63]]
precision of testing set: 0.7682926829268293
recall of testing set 0.6774193548387096
accuracy of testing set 0.6259541984732825
f1 of testing set 0.7199999999999999

#### ICA + LR
With ICA= 35 and l1, the best params are:
{'C': 1000} for n_compo= 35
[[147  69]
 [ 56 160]]
[[10 28]
 [32 61]]
precision of testing set: 0.6853932584269663
recall of testing set 0.6559139784946236
accuracy of testing set 0.7106481481481481
f1 of testing set 0.7191011235955055

With ICA= 40 and l2, the best params are:
{'C': 1000} for n_compo= 40
LogisticRegression(C=1000, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='saga', tol=0.1, verbose=0,
                   warm_start=False)
[[152  64]
 [ 65 151]]
[[12 26]
 [35 58]]
precision of testing set: 0.6904761904761905
recall of testing set 0.6236559139784946
accuracy of testing set 0.7013888888888888
f1 of testing set 0.7006960556844548
