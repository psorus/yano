import numpy as np

import yano
from yano.symbols import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *
from yano.logging import Logger

from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM as SVM
from pyod.models.lof import LOF
from pyod.models.pca import PCA

import json

from plt import *

clf=IForest(n_estimators=100)
#clf=negate_output(clf)
clf2=KNN(n_neighbors=3)
clf3=SVM()
clf4=LOF()
clf5=PCA()

log=Logger({"IFor":clf,"KNN":clf2,"SVM":clf3,"LOF":clf4,"PCA":clf5}, verbose=10)
log=Logger({"IFor":clf,"KNN":clf2,"LOF":clf4}, verbose=10)

condition=(number_of_samples>100) & (number_of_features<200) & (number_of_features>10)
#condition=condition & (numeric & ~(categorical | nominal | textual))
#condition=condition & ((nominal & numeric) & ~(categorical  | textual))
condition=condition & (nominal & ~(categorical  | textual | numeric))
condition=condition & ~index


print(f"found {len(condition)} datasets")

for d,x,tx,ty in pipeline(condition, split,nonconst, shuffle, normalize_minmax):
    log.run_on(d,x,tx,ty)

log.save("log.json")



#exit()

#likelys=[]

#for d,folds in pipeline(condition,nonconst,crossval,shuffle,normalize_minmax):
#for d,folds in pipeline(condition,crossval(9),normalize("none"),shuffle):
#    print(d)
#    stats=[]
#    for x,tx,ty in folds:
#        auc=test_algo_n(clf,x,tx,ty)
#        stats.append(auc)
#        print(auc)

#        if len(stats)>1:
#            break
#    lik=stats[0]==stats[1]
#    likelys.append(lik)
#    print(lik)

#plt.hist(np.log(likelys))
#plt.show()
    



