import numpy as np

import yano
from yano.symbols import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *


from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN

import json


condition=numeric & (~ nominal) & (~ textual) & (~ categorical)

condition=condition & (number_of_samples>3000)

#condition=logical_true

#condition=~numerical


condition=~index
condition=condition & (number_of_samples>2000)
condition=condition & (number_of_samples>20*number_of_features)
condition=condition & (number_of_features>10)
condition=condition & (number_of_features<100)

condition=condition & (name!="cifar0")

print(len(condition))

for d in condition:
    print(d)

exit()


pth="/home/psorus/useNikitha/"

count=0
for d,x,tx,ty in pipeline(condition, nonconst, shuffle, split):
    print(d)
    np.savez_compressed(pth+str(d)+".npz", x=x, tx=tx, ty=ty)
    count+=1

print()
print(count)


