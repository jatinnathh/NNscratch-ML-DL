import numpy as np


def euclidean_distance(x1,x2):
    dist=np.sqrt(np.sum((x1-x2)**2))
    return dist

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def cosine_similarity(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    if norm_x1 == 0 or norm_x2 == 0:
        return 0
    return dot_product / (norm_x1 * norm_x2)

def jaccard_similarity(x1, x2):
    intersection = np.sum(np.minimum(x1, x2))
    union = np.sum(np.maximum(x1, x2))
    if union == 0:
        return 0
    return intersection / union

def minkowski_distance(x1, x2, p=3):
    if p <= 0:
        raise ValueError("p must be greater than 0")
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)


def chebyshev_distance(x1, x2):
    pass
