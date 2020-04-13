import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

'''
Assert: dataset passed in is already ascaled
Assert: clf passed in is already trained
Assert: clf_dict is a dict of pipelines dict, as for each type of pipeline there
        should be 5 different models passed in
'''

'''
Format of input clf_dict:
clf_dict = {
    clf_1_name:{
        pca_clf_1:{
            0:{},
            1:{},
            2:{},
            3:{},
            4:{}
        },
        ica_clf_1:{...},
        umap_clf_1:{...},
        fs_clf_1:{...},
        vae_clf_1:{...},
    },
    clf_2_name:{
        pca_clf_2:{...},
        ica_clf_2:{...},
        ...
    },
    ...
}
'''
    
'''
overall_metrics_scores = {
    clf_1_name:{
        pca_clf_1_scores = {
            "acc":[.., .., .., .., ..],
            "prec":[.., .., .., .., ..],
            ...
        },
        ica_clf_1_scores = {...},
        ...
    },
    clf_2_name:{
        pca_clf_2_scores = {
            "acc":[.., .., .., .., ..],
            "prec":[.., .., .., .., ..],
            ...
        },
        ica_clf_2_scores = {...},
    }
}
'''
def eval(clf_dict, X, y):
    # final selected pipelines' metrics scores
    overall_metrics_scores = get_metrics_scores(clf_dict, X, y)
    print(overall_metrics_scores)
    final_ppls = get_final_ppls(overall_metrics_scores, clf_dict)
    print("The final pipelines and their metrics for each classifier are:")
    print(final_ppls)
    
    # one plot for all aucroc curve
    get_aucroc_plot(final_ppls, X, y)
    # plots for confusion matrix
    get_confmat_plots(final_ppls, X, y)
    

# def get_sigtest(clf_dict, X, y):
    
'''
get_impr_to_base: only compare the final 4 pipelines
overall_base_impr = {
    clf1:{final_ptype:{"acc_impr":[..,..,..,..,..],
                       "prec_impr":[..,..,..,..,..],
    }},
    clf2...
}
'''
def get_baseline_metrics():
    baseline_y = [1 for i in range(len(y))]
    prec_baseline = metrics.precision_score(y, baseline_y)
    acc_baseline = metrics.accuracy_score(y, baseline_y)
    auc_baseline = metrics.roc_auc_score(y, baseline_y)
    recall_baseline = metrics.recall_score(y, baseline_y) 
    f1_baseline = metrics.f1_score(y, baseline_y)

    baseline_eval_res = {
        "acc":[acc_baseline],
        "prec":[prec_baseline],
        "recall":[recall_baseline],
        "f1":[f1_baseline],
        "auc":[auc_baseline],
    }
    
    return baseline_eval_res
    
def get_impr_to_base(final_ppls):    # Only output accuracy and precision
    baseline_metrics = get_baseline_metrics()
    base_acc = np.array(baseline_metrics["acc"])
    base_prec = np.array(baseline_metrics["prec"])
    overall_base_impr = {}
    
    for key in final_ppls:
        cur_clf_type = key #String
        metrics = final_ppls[cur_clf_type]["metrics"]
        ptype = final_ppls[cur_clf_type]["ptype"] #String
        overall_base_impr[cur_clf_type] = {}
        
        # difference between the base acc and the current type's classifier's final pipeline's acc
        overall_base_impr[cur_clf_type][ptype]["acc_impr"]= np.subtract(np.array(metrics["acc"]), base_acc)
        overall_base_impr[cur_clf_type][ptype]["prec_impr"] = np.subtract(np.array(metrics["prec"]), base_prec)

    
    return overall_base_impr
    

'''
get_aucroc: only the final 4 pipelines
'''
def get_aucroc_plot(final_ppls, X, y):
    color_list = plt.cm.tab20(np.linspace(0, 1, 20))
    
    baseline_y = [1 for i in range(len(y))]
    curve_fig = plt.figure()
    ax1 = curve_fig.add_subplot(111)
    fpr, tpr, thresholds = metrics.roc_curve(y, baseline_y, pos_label=2)
    ax1.plot(fpr, tpr, linestyle="--", linewidth=2, color="black", label="baseline, auc=0.5")
    ax1.set_xlabel("False positive rate (fpr)")
    ax1.set_ylabel("True positive rate (tpr)")
    
    idx = 0
    for key in final_ppls:
        cur_clf_dict = final_ppls[key]
        ptype = cur_clf_dict["ptype"]
        clf = cur_clf_dict["pipeline"]
        y_pred = clf.predict(X)
        auc = metrics.roc_auc_score(y, y_pred)
        # Plot roc_auc curve
        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=2)
        ax1.plot(fpr, tpr, linestyle="-", linewidth=2, c=color_list[idx], label=ptype+",auc="+str(auc))
        idx = idx+1

    # Plot final baseline curve
    ax1.legend(bbox_to_anchor=(1.1, 1))
    curve_fig.savefig("../plotting/plots/auc-roc.png")
    plt.show()
    

'''
get_confmat_plots: only the final 4 pipelines
'''
def get_confmat_plots(final_ppls, X, y):
     for key in final_ppls:
        cur_clf_dict = final_ppls[key]
        ptype = cur_clf_dict["ptype"]
        clf = cur_clf_dict["pipeline"]
        # Plot confusion matrix
        disp = metrics.plot_confusion_matrix(clf, X, y,
                                 display_labels=["Health Control", "Parkinson's Disease"],
                                 cmap=plt.cm.GnBu,
                                 normalize="true")
        
        disp.ax_.set_title("Normalised confusion matrix for "+ptype)

        
        plt.savefig("../plotting/plots/"+key+"-conf-mat.png")
        plt.show()

        
        
        
        

#---------------
#Get metrics for all pipelines 
#---------------
def get_metrics_scores(clf_dict, X, y):
    #Construct basedline prediction as the comparison
    #For PPMI only
    # Predict everything to 1: PD
    overall_metrics_scores = {}
    for key in clf_dict:
        # Identify the classifier
        cur_clf_type = key
        overall_metrics_scores[cur_clf_type] = {}
        #cur_clf_type_dict: the pipeline dict for each type of classifier
        cur_clf_type_dict = clf_dict[cur_clf_type]

        # Identify the pipeline type of the required classifier
        for key in cur_clf_type_dict:
            cur_ptype = key
            overall_metrics_scores[cur_clf_type][cur_ptype] = {
                "acc":[],
                "prec":[],
                "recall":[],
                "f1":[],
                "auc":[]
            }
            
            # Identify the actual pipeline model of the pipeline type
            cur_clf_dict = cur_clf_type_dict[cur_ptype] #clf = ppl here
            for key in cur_clf_dict:
                #clf: each model of the same type of pipeline
                clf = cur_clf_dict[key]
                y_pred = clf.predict(X)
                overall_metrics_scores[cur_clf_type][cur_ptype]["acc"].append(metrics.accuracy_score(y, y_pred))
                overall_metrics_scores[cur_clf_type][cur_ptype]["prec"].append(metrics.precision_score(y, y_pred))
                overall_metrics_scores[cur_clf_type][cur_ptype]["recall"].append(metrics.recall_score(y, y_pred))
                overall_metrics_scores[cur_clf_type][cur_ptype]["f1"].append(metrics.f1_score(y, y_pred))
                overall_metrics_scores[cur_clf_type][cur_ptype]["auc"].append(metrics.roc_auc_score(y, y_pred))
            
    return overall_metrics_scores


#---------------
# Get pipeline type name, pipeline object and the metrics of the final selected pipelines
#---------------

def get_final_ppls(overall_metrics_scores, clf_dict): #return both the final selected pipeline                                              and its metrics
    final_ppls = {}
    for key in overall_metrics_scores:     # identify current classifier type
        cur_clf_type = key #key:String
        cur_ptype_dict = overall_metrics_scores[cur_clf_type] #value:dict
        final_ppls[cur_clf_type] = {
            "ptype": None, #String
            "pipeline":None, #Pipeline object
            "metrics":{} #metrics dict
        } # dict to store the final ppl object and its metrics
        
        max_mean_prec = 0
        final_ptype = None

        ## 1. find out which ptype contains the best pipeline object
        for key in cur_ptype_dict:         # identify current pipeline type
                                           # key:String
            cur_ptype_mean_prec = np.mean(np.array(cur_ptype_dict[key]["prec"]))  #find the ptype with the highest mean prec
            if cur_ptype_mean_prec > max_mean_prec:
                max_mean_prec = cur_ptype_mean_prec
                final_ptype = key
        
        ## 2. find out the best pipeline object under the type
        ppl_id = get_max_prec_pos(cur_ptype_dict[final_ptype]["prec"])
        
        final_ppls[cur_clf_type]["ptype"] = final_ptype
        final_ppls[cur_clf_type]["pipeline"] = clf_dict[cur_clf_type][final_ptype][ppl_id]
        final_ppls[cur_clf_type]["metrics"] = overall_metrics_scores[cur_clf_type][final_ptype] 
        
    return final_ppls

#----------helpers---------------

def get_max_prec_pos(precarray):
    max = 0
    pos = 0
    for i in range(len(precarray)):
        if precarray[i] > max:
            max = precarray[i]
            pos = i
    return pos