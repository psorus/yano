import numpy as np

def seed_once(func, seed=42):

    def wrapper(*args,seed=0,base_seed=seed, **kwargs):
        post=np.random.randint(0,2**32-1)
        np.random.seed(base_seed+seed)
        result = func(*args, **kwargs)
        np.random.seed(post)
        return result
    return wrapper


