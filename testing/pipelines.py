from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.decomposition import FastICA
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.pipeline import Pipeline

'''
Assert: dr can only be one of the following:
- pca, ica, umap, fs
Assert: clf can only be one of the following:
- lr, svm, xgb

NO MLP OR VAE HERE

params = {
    "pca": {"n":n},
    "ica": {"n":n},
    "umap":{"n_neighbours":k, "min_dist":md, "n":n},
    "fs":{"a":a},
    "lr":{"C":C, "reg":reg},
    "svm:"{"kernel", kernel, "C":C, "gamma":gamma, "coef0":c, "degree":d},
    "xgb":{"n":n, "h":h, "lr":lr, "s":s, "c":c}
}
'''
def get_pipeline(dr_name, clf_name, params):
    if dr_name == "pca":
        dr = PCA(n_components=params["pca"]["n"])
    elif dr_name == "ica":
        dr = FastICA(n_components=params["ica"]["n"])
    elif dr_name == "umap":
        dr = UMAP(
            n_components=params["umap"]["n"], 
            n_neighbours=params["umap"]["n_neighbours"],
            min_dist=params["umap"]["min_dist"],
            max_depth=params["umap"]["min_dist"],
        )
    elif dr_name == "fs":
        dr = SelectFromModel(Lasso(alpha=params["fs"]["a"], tol=0.01, random_state=42))
    
    if clf_name == "lr":
        clf = LogisticRegression(C=params["lr"]["C"], penalty=params["lr"]["reg"], max_iter=1000, solver='saga', tol=0.1)
    elif clf_name == "svm":
        clf = SVC(max_iter=3000, gamma=params["svm"]["gamma"], kernel=params["svm"]["kernel"], coef0=params["svm"]["coef0"], C=params["svm"]["C"], degree=params["svm"]["degree"],tol=0.01,class_weight='balanced')
    elif clf_name == "xgb":
        clf = xgb_clf = xgb.XGBClassifier(
                    objective='binary:logistic', 
                    seed=42, 
                    tree_method='gpu_hist',
                    learning_rate=params["xgb"]["lr"],
                    subsample=params["xgb"]["s"],
                    gpu_id=1,
                    colsample_bytree=params["xgb"]["c"],
                    n_estimators=params["xgb"]["n"],
                    max_depth=params["xgb"]["h"]
                )
    
    
    pl = Pipeline([
    ("dr", dr),
    ("clf", clf)
])
    return pl
        
        
        