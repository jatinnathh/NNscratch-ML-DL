import numpy as np


def euclidean_distance(x1,x2):
    dist=np.sqrt(np.sum((x1-x2)**2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))
