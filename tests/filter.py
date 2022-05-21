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

count=0
for zw in condition:
    print(zw)
    count+=1

print()
print(count)


