import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.decomposition import FastICA
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import LinearSVC, SVC
import xgboost as xgb

from sklearn.pipeline import Pipeline

'''
Assert: the pipeline returned is already trained
'''
def get_default_clf_dict(X_train, y_train, gpu_id):
    default_params_clf = {
        "lr":{},
        "svm":{},
        "xgb":{},
    }
    for clf_name in ["lr", "svm", "xgb"]:
        for dr_name in ["pca", "ica", "umap"]:  
            print ("Current pipeline is:", dr_name + " " + clf_name)
            if dr_name == "pca":
                dr = PCA()
            elif dr_name == "ica":
                dr = FastICA()
            elif dr_name == "umap":
                dr = UMAP()
#             elif dr_name == "fs":
#                 dr = SelectFromModel(Lasso(alpha=params["fs"]["a"], tol=0.01, random_state=42))

            if clf_name == "lr":
                clf = LogisticRegression()
            elif clf_name == "svm":
                clf = SVC()
        
            cur_pipeline = Pipeline(steps=[
                ("dr", dr),
                ("clf", clf)
            ])
            cur_pipeline.fit(X_train, y_train)
            default_params_clf[clf_name][dr_name+"_"+clf_name+"_base"] = {}
            default_params_clf[clf_name][dr_name+"_"+clf_name+"_base"][0] = cur_pipeline
    
    return default_params_clf