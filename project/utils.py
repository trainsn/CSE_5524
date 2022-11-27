import numpy as np

def l1_normalize(v):
    norm = abs(v).sum()
    if norm == 0:
       return v
    return v / norm

def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm
