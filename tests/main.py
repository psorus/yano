import numpy as np

import yano
from yano.meta import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *


from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN

clf=IsolationForest(n_estimators=100)#,random_state=42)
#clf=KNN(n_neighbors=10)


try:
    condition=number_of_features & number_of_samples
    condition=~condition
except:
    condition=ltrue

#for d,x,tx,ty in pipeline(condition, split,nonconst, shuffle, normalize_minmax):
#    print(d,test_algo(clf,x,tx,ty))

#exit()

#for d,folds in pipeline(condition,nonconst,crossval,shuffle,normalize_minmax):
for d,folds in pipeline(condition,crossval):
    print(d)
    stats=[]
    for x,tx,ty in folds:
        auc=test_algo_n(clf,x,tx,ty)
        stats.append(auc)
        print(auc)

mat=similarity_matrix(stats)
print(mat)
mat=np.array(to_likelihood(mat))
print(mat)

sol=combine_stats(*stats)
print(sol)

for stat in stats:
    print(is_same(stat,sol))




