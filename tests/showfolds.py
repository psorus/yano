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

for d,folds in pipeline(name=="cardio", shuffle, normalize("minmax"), crossval(3)):break

for d,px,py in pipeline(name=="cardio", normalize("minmax")):break

folds=list(folds)


funcs=run_algo_folds(clf,folds)

aucs=[]
for func,(x,tx,ty) in zip(funcs,folds):
    auc=eval_func(func, tx,ty)
    aucs.append(auc)

auc=Stats(aucs)

altitude=0.0
inc=0.25

px0=np.array([xx for xx,yy in zip(px,py) if yy==0])

altitude_plot(funcs, px0, px, py, altitude=0, inc=inc, hist=True)

plt.how()

exit()

altitude_plot(None, px,px,py, inc=inc)

for i,(func,(x,tx,ty)) in enumerate(zip(funcs,folds)):
    altitude_plot(func, x, tx, ty, altitude=altitude, inc=inc, hist=False)

plt.how()








