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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.decomposition import FastICA

from sklearn.feature_selection import SelectFromModel


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
    
# split.get_n_splits(X_train, y_train)
# for train_index, val_index in split.split(X_train, y_train):
#     print("TRAIN:", len(train_index), "VALIDATE:", len(val_index))
#     X_train, X_val = X_train[train_index], X_train[val_index]
#     y_train, y_val = y_train[train_index], y_train[val_index]

    
    
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kernels = ['rbf', 'poly', 'linear', 'sigmoid']
# C_options=[0.01, 1, 1000]
# gamma=[1e-4, 0.01, 1, 1.5]

# # for k in kernels:
# #     for c in C:
# #         svm = SVC(C=c, kernel=k)
# #         svm.fit(X_train_scaled, y_train)
# #         y_pred_svm = svm.predict(X_val_scaled) 
# #         print('SVM with ', k, ' kernel, C value =', c, 'has accuracy: ', accuracy_score(y_val, y_pred_svm))



# param_grid = [
#     {
#         'C': C_options,
#         'kernel': kernels,
#         'gamma':gamma,
#     }
# ]

# grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="accuracy", n_jobs=3)
# grid.fit(X_train_scaled, y_train)

# mean_scores = np.array(grid.cv_results_['mean_test_score'])
# print(mean_scores)
# print('Best estimator: ', grid.best_params_)

# ####Documentation 
# # Best estimator:
# # C = 0.01
# # gamma = 0.0001
# # kernel = rfb

n_components = [50, 100, 150, 200, 250]
kernels = ['rbf', 'poly', 'linear', 'sigmoid']
C_options=[0.001, 0.01, 0.1, 1]

### PCA
# best_perf=0
# for n in n_components:
#     pca = PCA(n_components=n)
#     X_train = pca.fit_transform(X_train_scaled)
#     X_test = pca.transform(X_test_scaled)
#     print('Shape of PCs:', X_train.shape[1])
    
### UMAP
# best_perf=0
# for n in n_components:
#     umap = UMAP(n_components=n)
#     X_train = umap.fit_transform(X_train_scaled)
#     X_test = umap.transform(X_test_scaled)
#     print('Shape of UMAP clusters:', X_train.shape[1])
    
# # ### ICA
# best_perf=0
# for n in n_components:
#     ica = FastICA(n_components=n)
#     X_train = ica.fit_transform(X_train_scaled)
#     X_test = ica.transform(X_test_scaled)



#Lasso Reg for FS
alpha=[0.001,0.01, 0.1]
for a in alpha:
    sel_ = SelectFromModel(Lasso(alpha=a, tol=0.01, random_state=42))
    sel_.fit(X_train_scaled, y_train)
    X_train = sel_.transform(X_train_scaled)
#     X_test_selected = sel_.transform(X_test_scaled)
    print("Shape of training set with alpha=", a, ":", X_train.shape)
    
    
    best_perf=0
    for k in kernels:
        svc = SVC(max_iter=3000, gamma=0.001, kernel=k, tol=0.01)
        param_grid ={'C': C_options, }
        grid = GridSearchCV(svc, param_grid=param_grid, scoring="accuracy", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=3)
        grid.fit(X_train, y_train)
        # evaluation metric is accuray 

        print("SVM on FS Reg_alpha=", a, " and kernel ",k,":")
#         print("SVM on n_compo=", n, " and kernel ",k,":")
        print("Best accuracy:", grid.best_score_)
        print('Best estimator: ', grid.best_params_)
        print()
        if grid.best_score_ > best_perf:
            best_perf = grid.best_score_
            best_param = grid.best_params_
            kernel_flag = k
            a_flag = a
#             compo_flag = n
    
    print("SVM with Ref FS alpha=", a_flag, 'and kernel', kernel_flag, 'has best performance of',best_perf, "with", best_param)
#     print("SVM with PCs=", compo_flag, 'and kernel', kernel_flag, 'has best performance of',best_perf, "with", best_param)
#     print("SVM with UMAP clusters=", compo_flag, 'and kernel', kernel_flag, 'has best performance of',best_perf, "with", best_param)
    print()
    

    
########Documentation######
####PCA+SVM
# SVM on PCA pc= 50 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.01, 'kernel': 'rbf'}
# SVM on PCA pc= 100 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.01, 'kernel': 'rbf'}
# SVM on PCA pc= 150 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.01, 'kernel': 'rbf'}
# SVM on PCA pc= 200 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.01, 'kernel': 'rbf'}
# SVM on PCA pc= 250 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.01, 'kernel': 'rbf'}
###Showing same result with kernel='rbf', but an increase in the performance for other non-optimal kernels as n_PC increases
##Hence use PC = 250

#####UMAP+SVM
# SVM on UMAP clusters= 50 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.001, 'kernel': 'rbf'}
# SVM on UMAP clusters= 100 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.001, 'kernel': 'rbf'}
# SVM on UMAP clusters= 150 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.001, 'kernel': 'rbf'}
# SVM on UMAP clusters= 200 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.001, 'kernel': 'rbf'}
# SVM on UMAP clusters= 250 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.001, 'kernel': 'rbf'}
###Both kernels give 0.709.... accuracy

#####ICA+SVM
# SVM on ICA ic= 50 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.001, 'kernel': 'rbf'}
# SVM on ICA ic= 100 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.001, 'kernel': 'rbf'}
# SVM on ICA ic= 150 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.001, 'kernel': 'rbf'}
# SVM on ICA ic= 200 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.001, 'kernel': 'rbf'}
# SVM on ICA ic= 250 :
# Best accuracy: 0.7097701149425287
# Best estimator:  {'C': 0.001, 'kernel': 'rbf'}
###Showing same result with kernel='rbf', but an increase in the performance for other non-optimal kernels as n_PC increases
##Hence use PC = 250

######FS+SVM
# Shape of training set with C= 1 : (348, 132072)
# SVM with Ref FS C= 1 and kernel linear has best performance of 0.9396551724137931 with {'C': 0.001}

# Shape of training set with C= 10 : (348, 155839)
# VM with Ref FS C= 10 and kernel linear has best performance of 0.9367816091954023 with {'C': 0.001}

# Shape of training set with C= 100 : (348, 190133)
# SVM with Ref FS C= 100 and kernel linear has best performance of 0.9109195402298851 with {'C': 0.001}

# Shape of training set with C= 1000 : (348, 190701)
# SVM with Ref FS C= 100 and kernel linear has best performance of 0.9166666666666666 with {'C': 0.001}

###With Lasso
# Shape of training set with alpha= 0.001 : (348, 809)
# SVM with Ref FS alpha= 0.001 and kernel linear has best performance of 1.0 with {'C': 0.001}

# Shape of training set with alpha= 0.01 : (348, 323)
# SVM with Ref FS alpha= 0.01 and kernel linear has best performance of 1.0 with {'C': 0.001}
# This one has the highest accuracy for all four kernels compared to other two alpha values

# Shape of training set with alpha= 0.1 : (348, 4)
# SVM with Ref FS alpha= 0.1 and kernel linear has best performance of 0.7212643678160919 with {'C': 0.1}

