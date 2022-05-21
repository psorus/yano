import yano


from yano.symbols import *
from yano.vis import *
from yano.iter import *
from yano.utils import *


from plt import *

from pyod.models.iforest import IForest
from pyod.models.knn import KNN

import numpy as np


clf=IForest(n_estimators=100)
#clf=KNN(n_neighbors=5)

for d,x,tx,ty in pipeline(name=="cardio", split, shuffle, normalize("minmax")):break


func=run_algo(clf,x)

auc=eval_func(func,tx,ty)
print(auc)


p=func(tx)

hist(p,ty, modus="stack")
plt.show()

mn,mx=np.min(p),np.max(p)
mn,mx=mn+0.05*(mx-mn),mx-0.05*(mx-mn)
couop=5
outer=0
borders=np.linspace(mn,mx,couop+outer)
if outer>0:borders=borders[1:-1]
if outer>2:borders=borders[1:-1]

borders=list(borders)



altitude_plot(func,x, tx, ty, borders,inc=0.25)
plt.how()







