import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score

def multiclass_metrics(y_true, probabilities):
    y_true=np.asarray(y_true,dtype=int); probabilities=np.asarray(probabilities,dtype=float); y_pred=np.argmax(probabilities,axis=1); onehot=np.eye(probabilities.shape[1])[y_true]
    return {
        "accuracy":float(accuracy_score(y_true,y_pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
        "macro_f1":float(f1_score(y_true,y_pred,average="macro")),
        "macro_roc_auc_ovr":float(roc_auc_score(y_true,probabilities,average="macro",multi_class="ovr")),
        "macro_pr_auc":float(average_precision_score(onehot,probabilities,average="macro")),
    }
