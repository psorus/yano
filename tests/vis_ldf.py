import numpy as np

import yano
from yano.symbols import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *
from yano.vis import *

from plt import *

import os

from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN

import json


dset="phoneme"
for d,x,tx,ty in pipeline(name==dset,nonconst, split, shuffle, normalize("minmax")):break




clf=KNN(n_neighbors=5)

func=run_algo(clf,x)

auc=eval_func(func,tx,ty)
print(auc)

p=func(tx)
hist(p,ty,modus="stack")
plt.show()

mn,mx=np.min(p),np.max(p)
mn,mx=mn+0.05*(mx-mn),mx-0.05*(mx-mn)
couop=5
outer=0
borders=np.linspace(mn,mx,couop+outer)
if outer>0:borders=borders[1:-1]
if outer>2:borders=borders[1:-1]

borders=list(borders)

tx0=np.array([txx for txx,tyy in zip(tx,ty) if tyy==0])
ty0=np.array([tyy for tyy in ty if tyy==0])
tx1=np.array([txx for txx,tyy in zip(tx,ty) if tyy==1])
ty1=np.array([tyy for tyy in ty if tyy==1])



altitude_plot(func,x, tx, ty ,[],inc=0.25)
plt.how()
