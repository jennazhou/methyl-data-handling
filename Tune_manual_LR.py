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

from sklearn.decomposition import PCA
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
    
    
C_options = [0.01, 0.1, 1, 1.5, 10, 100]

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

n_components = [50, 100, 150, 200, 250]

pipe = Pipeline([
    ('pca', PCA()),
    ('clf', LogisticRegression(max_iter=500, penalty='l1', solver='saga'))
])

param_grid = [
    {
        'pca__n_components': n_components,
        'clf__C': C_options,
    },
]

grid = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=6)
grid.fit(X_train_scaled, y_train)
# evaluation metric is accuray 

mean_scores = np.array(grid.cv_results_['mean_test_score'])
print("With PCA and l1:")
print(grid.cv_results_['params'])
print(mean_scores)
print(grid.best_params_)



# ###L2
# pipe = Pipeline([
#     ('pca', PCA()),
#     ('clf', LogisticRegression(max_iter=500, penalty='l2', solver='saga'))
# ])

# param_grid = [
#     {
#         'pca__n_components': n_components,
#         'clf__C': C_options,
#     },
# ]

# grid = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=6)
# grid.fit(X_train_scaled, y_train)
# # evaluation metric is accuray 

# mean_scores = np.array(grid.cv_results_['mean_test_score'])
# print("With PCA and l2:")
# print(grid.cv_results_['params'])
# print(mean_scores)
# print(grid.best_params_)


#######################Mannual Tuning Data: Need to redo#############
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
