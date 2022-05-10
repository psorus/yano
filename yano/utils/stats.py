import json
import numpy as np



class stats(object):
    def __init__(self,mean=None,std=None,mx=None,mn=None,n=None):
        if type(mean) is list:
            std=np.std(mean)
            mx=np.max(mean)
            mn=np.min(mean)
            n=len(mean)
            mean=np.mean(mean)
        self.stats={}
        if not mean is None:self.stats['mean']=mean
        if not std is None:self.stats['std']=std
        if not mx is None:self.stats['max']=mx
        if not mn is None:self.stats['min']=mn
        if not n is None:self.stats['n']=n

    def __getitem__(self,key):
        if not key in self.stats:
            raise KeyError("This object did not get enough information to compute the requested statistic: "+key)
        return self.stats[key]


    def __str__(self):
        if not "mean" in self.stats:
            return ""
        else:
            ret=str(self.stats['mean'])
        if "std" in self.stats:
            ret+=" +- "+str(self.stats['std'])
        if "max" in self.stats and "min" in self.stats:
            ret+=" ["+str(self.stats["min"])+" - "+str(self.stats["max"])+"]"
        elif "max" in self.stats:
            ret+=" [? - "+str(self.stats["max"])+"]"
        elif "min" in self.stats:
            ret+=" ["+str(self.stats["min"])+" - ?]"
        if "n" in self.stats:
            ret+=" (n="+str(self.stats["n"])+")"
        return ret

    def __repr__(self):
        return json.dumps(self.stats)

    def __float__(self):
        assert "mean" in self.stats
        return float(self.stats['mean'])


def combine(*q):
    """
    Combine stats objects
    """
    sums=0.0
    mx=float(q[0])
    mn=float(q[0])
    n=0

    stds=[]

    for qq in q:
        if "n" in qq.stats:
            n+=qq.stats['n']
            sums+=qq.stats['n']*qq.stats['mean']
            if "std" in qq.stats:
                #yeah, this is a bit ugly
                for zw in np.random.normal(qq.stats['mean'],qq.stats['std'],int(qq.stats['n'])):
                    stds.append(zw)
            else:
                for i in range(qq.stats["n"]):
                    stds.append(qq.stats["mean"])
        else:
            n+=1
            sums+=qq.stats['mean']
            if "std" in qq.stats:
                stds.append(np.random.norma(qq.stats['mean'],qq.stats['std']))
            else:
                stds.append(qq.stats['mean'])
        if "max" in qq.stats:
            mx=max([mx,qq.stats['max']])
        else:
            mx=max([mx,qq.stats['mean']])
        if "min" in qq.stats:
            mn=min([mn,qq.stats['min']])
        else:
            mn=min([mn,qq.stats['mean']])

    std=None
    if len(stds)>0:
        std=np.std(stds)

            




    return stats(mean=sums/n,std=std,mx=mx,mn=mn,n=n)
        






