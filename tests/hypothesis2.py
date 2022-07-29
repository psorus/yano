import numpy as np

import yano
from yano.symbols import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *
from yano.logging import Logger
from yano.helper import nannable

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

import eif_old as iso

import json

#choose datasets
condition=~index#let each dataset be uncorrelated to the other datasets
condition=condition & (number_of_samples>200)
condition=condition & (number_of_features>50)

condition=condition & (number_of_features<500)
condition=condition & (number_of_samples<10000)
condition=condition & name
#condition=condition & (numeric & nominal)


print(len(condition), "Datasets found")#How many datasets are left?

exit()

#define hypothesis
clf1=IForest(n_estimators=100)
clf2=LOF()
clf3=KNN(n_neighbors=5)

l=Logger({"IFor":clf1,"Lof":clf2, "Knn":clf3}, verbose=10, addfeat=True)

if True:

    
    for dataset, folds in pipeline(condition, crossval(5), normalize("minmax"), shuffle):
        folds=list(folds)
        l.run_cross(dataset, folds)


   

    l.save("tex/highdim_nn.json")

else:
    l.load("tex/highdim_nn.json")

l.show()
texouter='''
\\documentclass{article}
\\usepackage[utf8]{inputenc}


\\begin{document}


###




\\end{document}

'''

with open("tex/highdim_nn.tex", "w") as f:
    f.write(texouter.replace("###",l.to_latex()))
    



