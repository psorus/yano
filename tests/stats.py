import numpy as np

import yano
from yano.meta import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *


from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN
from pyod.models.iforest import IForest


import json

from plt import *

clf=IsolationForest(n_estimators=100)#,random_state=42)
clf=negate_output(clf)
#clf=KNN(n_neighbors=3)
#clf=IForest(n_estimators=100)



try:
    condition=number_of_features & number_of_samples
    condition=condition
except:
    condition=ltrue

condition=(number_of_samples>100) & (number_of_features<100) & (number_of_features>10)
condition=condition & (numeric & ~(categorical | nominal | textual))


print(f"found {len(condition)} datasets")

#for d,x,tx,ty in pipeline(condition, split,nonconst, shuffle, normalize_minmax):
#    print(d,test_algo(clf,x,tx,ty))

#exit()

likelys=[]

#for d,folds in pipeline(condition,nonconst,crossval,shuffle,normalize_minmax):
for d,folds in pipeline(condition,crossval(9),normalize("none"),shuffle):
    print(d)
    stats=[]
    for x,tx,ty in folds:
        auc=test_algo_n(clf,x,tx,ty)
        stats.append(auc)
        print(auc)

        if len(stats)>1:
            break
    lik=stats[0]==stats[1]
    likelys.append(lik)
    print(lik)

plt.hist(np.log(likelys))
plt.show()
    



