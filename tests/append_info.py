import numpy as np

import yano
from yano.symbols import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *

import os

from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN

import json


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


fn="tex/ldf.json"
l=Logger({},verbose=10)
l.load(fn)

def nevercall(*args,**kwargs):
    raise Exception("Never call this")

add={"Abalone_1_8":0.213275757507693,
"banknote-authentication":0.007521545090797172,
"cardio":0.05152376033057852,
"MagicTelescope":0.13395600003244998,
"mammography":0.1296671597633136,
"page-blocks":0.0668048469387755,
"pendigits":0.0014792899408284,
"phoneme":0.2383066684464446,
"pollen":0.5131003907740088,
"satellite":0.20921034555895895,
"segment":0.0009182736455463687,
"steel-plates-fault":0.2672222222222222,
"wbc":0.009070294784580504}

add={key:1-value for key,value in add.items()}


l.add_algo("ldf",nevercall, add)
l.show(addfeat=True)

