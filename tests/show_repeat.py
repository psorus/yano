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

for d,x,tx,ty in pipeline(name=="cardio", shuffle, normalize("minmax"), split):break

for d,px,py in pipeline(name=="cardio", normalize("minmax")):break



funcs=run_algo_n(clf,x, n=3)

altitude=0.0
inc=0.25

px0=np.array([xx for xx,yy in zip(px,py) if yy==0])

altitude_plot(funcs, px0, px, py, altitude=0, inc=inc, hist=True)

plt.how()







