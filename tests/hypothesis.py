import numpy as np

import yano
from yano.symbols import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *
from yano.logging import Logger
from yano.helper import nannable

from pyod.models.iforest import IForest
import eif_old as iso

import json

#choose datasets
condition=~index#let each dataset be uncorrelated to the other datasets
condition=condition & (number_of_samples>200)#only use datasets with more than 200 samples
condition=condition & (number_of_samples>2*number_of_features)#In case there are many features, require more samples
condition=condition & (number_of_features>5)#only use datasets with more than 5 features

condition=condition & (number_of_features<100)#only use datasets with less than 100 features
condition=condition & (number_of_samples<10000)#only use datasets with less than 10000 samples


#condition= condition & ~((name=="pc3") | (name=="ozone-level-8hr"))#remove a single dataset as this makes problems

print(len(condition), "Datasets found")#How many datasets are left?


#define hypothesis
clf1=IForest(n_estimators=100)

def train_extended_ifor(x):
    clf=iso.iForest(x, ntrees=100, sample_size=min([256,len(x)]), ExtensionLevel=1)
    return nannable(clf.compute_paths)#not the best implementation, so in case something fails, return nan. For the comparison Nans are skipped


l=Logger({"IFor":clf1,"eIFor":train_extended_ifor}, verbose=10)

if True:

    
    for dataset, folds in pipeline(condition, crossval(5), normalize("minmax"), shuffle):
        folds=list(folds)
        l.run_cross(dataset, folds)
    
    l.save("tex/eifor_actual2.json")

else:
    l.load("tex/eifor_actual2.json")

l.show()
texouter='''
\\documentclass{article}
\\usepackage[utf8]{inputenc}


\\begin{document}


###




\\end{document}

'''

with open("tex/eifor_actual2.tex", "w") as f:
    f.write(texouter.replace("###",l.to_latex()))
    



