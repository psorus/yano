import numpy as np

import yano
from yano.symbols import *#logical_true as ltrue
from yano.iter import *
from yano.utils import *
from yano.logging import Logger

from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN
from pyod.models.iforest import IForest


import json

from plt import *

log=Logger({}, addfeat=True)

log.load("log.json")


log.show()






texouter='''
\\documentclass{article}
\\usepackage[utf8]{inputenc}


\\begin{document}


###




\\end{document}

'''










with open("tex/main.tex","w") as f:
    f.write(texouter.replace("###",log.to_latex()))



