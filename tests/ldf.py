import numpy as np

import yano
from yano.symbols import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *

import os

from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN

import json


condition=~index
condition=condition & (number_of_samples>1000)
condition=condition & (number_of_samples>200*number_of_features)
condition=condition & (number_of_features<50)
condition=condition & (numeric)

other=["cardio","segment","steel-plates-fault","wbc","satellite","qsar-biodeg","gas-drift","har","mnist"]

for oth in other:
    condition=condition | (name==oth)

print(len(condition))

pth="/home/psorus/useldf/"

count=0
for d,x,tx,ty in pipeline(condition, nonconst, shuffle, split, normalize("minmax")):
    print(d)
    np.savez_compressed(pth+str(d)+".npz", x=x, tx=tx, ty=ty)
    count+=1

print()
print(count)


from yano.logging import Logger
from yano.helper import nannable



from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder

def train_ae(x):
    dim=int(x.shape[1])
    ae=AutoEncoder(hidden_neurons=[dim//2,dim//3,dim//2])
    ae.fit(x)
    return ae.decision_function


l=Logger({"IForest":IForest(), "KNN":KNN(n_neighbors=5), "LOF":LOF(), "AE":train_ae}, verbose=10)
#l=Logger({"IForest":IForest(), "KNN":KNN(n_neighbors=5), "LOF":LOF(), "AutoEncoder":AutoEncoder()}, verbose=10)

fn="tex/ldf.json"
allow_load=True

if os.path.isfile(fn) and allow_load:
    l.load(fn)
else:
    for dataset, x,tx,ty in pipeline(condition, nonconst, shuffle, split, normalize("minmax")):
        l.run_on(dataset, x, tx, ty)

    l.save(fn)

l.show(addfeat=True)




