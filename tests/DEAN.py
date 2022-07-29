import yano


from yano.symbols import *
from yano.vis import *
from yano.iter import *
from yano.utils import *

import numpy as np

from plt import *


for d,x,tx,ty in pipeline(name=="satellite", split, shuffle, normalize("minmax")):break


from yano.ensemble import DEAN
from yano.ensemble import RandNet

#d=DEAN("satellite", x,tx,ty,bag=20)
d=RandNet("satellite", x,tx,ty)




scores,aucs=d.train_many(0,10)

plt.plot(aucs)
plt.show()







