import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


### Tune n_components for PCA+SVM

n_components = [50, 100, 150, 200, 250]
kernels = ['rbf', 'poly', 'linear', 'sigmoid']
C_options=[0.01, 1, 1000]

pipe = Pipeline([
    ('pca', PCA()),
    ('clf', SVC(gamma=0.0001))
])

param_grid = [
    {
        'pca__n_components': n_components,
        'clf__C': C_options,
        'clf__kernel': kernels,
    },
]

grid = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=5)
grid.fit(X_train_scaled, y_train)
# evaluation metric is accuray 

mean_scores = np.array(grid.cv_results_['mean_test_score'])
print(grid.cv_results_['params'])
print(mean_scores)
print('Best estimator: ', grid.best_params_)


