from yano.quick import quickcardio
from yano.utils import eval_func, roc_auc_score

import eif_old as iso

import numpy as np

x,tx,ty=quickcardio()


def train_extended_ifor(x):
    clf=iso.iForest(x.astype(np.float64), ntrees=100, sample_size=256, ExtensionLevel=1)
    return clf.compute_paths

func=train_extended_ifor(x)

auc=eval_func(func, tx, ty)

print(auc)




