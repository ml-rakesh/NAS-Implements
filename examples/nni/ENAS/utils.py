import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# Some metrics function
def accuracy_metrics(logits, y_true):
    return {'acc_score': accuracy(logits,y_true), 'f1_score': f1(logits,y_true)}

# accuracy function
def accuracy(logits,y_true):
    logits = logits.detach()
    score = accuracy_score(y_true,np.argmax(logits,axis=1))
    return score

# f1 score function
def f1(logits,y_true):
    logits = logits.detach()
    score = f1_score(y_true,np.argmax(logits,axis=1))
    return score