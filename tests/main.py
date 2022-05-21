import numpy as np

import yano
from yano.symbols import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *


from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN

import json

clf=IsolationForest(n_estimators=100)#,random_state=42)
#clf=KNN(n_neighbors=3)


try:
    condition=number_of_features & number_of_samples
    condition=~condition
except:
    condition=ltrue

#for d,x,tx,ty in pipeline(condition, split,nonconst, shuffle, normalize_minmax):
#    print(d,test_algo(clf,x,tx,ty))

#exit()

#for d,folds in pipeline(condition,nonconst,crossval,shuffle,normalize_minmax):
for d,folds in pipeline(condition,crossval(9),normalize("none"),shuffle(seed=41)):
    print(d)
    stats=[]
    for x,tx,ty in folds:
        auc=test_algo_n(clf,x,tx,ty)
        stats.append(auc)
        print(auc)

    with open("last.json","w") as f:
        json.dump([zw.q for zw in stats],f,indent=2)



#mat=similarity_matrix(stats)
#print(mat)
#print("!",np.mean(mat))
#mat=np.array(to_likelihood(mat))
#print(mat)

print()

sol=combine_stats(*stats)
print(sol)

#alls=[]
#for stat in stats:
#    print(zw:=is_same(stat,sol))
#    alls.append(zw)
#print("!",np.mean(alls))




