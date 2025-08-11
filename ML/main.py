import  numpy as np
import pandas as pd
from collections import Counter
from typing import Callable, Union
from stats import euclidean_distance as euc_dist 
from stats import manhattan_distance 



class KNN:

    def __init__(self,k=3, metric: Union[str, Callable] = 'euclidean',weighted=False,task="classification",return_proba=False):
        """
        k: number of neighbors
        metric: 'euclidean', 'manhattan', or a custom callable
        weighted: True -> weight votes by inverse distance
        task: 'classification' or 'regression'
        return_proba: if True, returns probabilities for classification
        """
         
        self.k=k
        self.metric=metric
        self.weighted=weighted
        self.task=task
        self.return_proba=return_proba


        if callable(metric):
            self.distance_function = metric
        elif metric=='euclidean':
            self.distance_function = euc_dist
        elif metric=='manhattan':
            self.distance_function = manhattan_distance
        else:
            raise ValueError("Metric must be 'euclidean', 'manhattan', or a callable function.")
        
    def fit(self,X,y):
        x=np.array(X)
        y=np.array(y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if self.k > len(X):
            raise ValueError(f"k={self.k} cannot be greater than number of training samples={len(X)}.")

        self.X_train = X
        self.y_train = y

    def predict(self,X):
        predictions=[self._predict_one(x) for x in X]
        return predictions

    def _predict_one(self,x):
        dist=np.array([self.distance_function(x,x_train) for x_train in self.X_train])

        #  returns the indices that would sort dist in ascending order (closest first) .
        k_indices=np.argsort(dist)[:self.k]

        #  takes the first K indices
        k_labels=self.y_train[k_indices]
        k_distances=dist[k_indices]

        if self.task=="classification":
            if self.weighted:
                """ 
                k_labels → The predicted labels of the k nearest neighbors.
                Example: ['cat', 'cat', 'dog']

                weights → The influence of each neighbor (often based on distance).
                Example: [0.8, 0.6, 0.3] (closer points → higher weight)

                class_weights → A dictionary to store the total vote weight for each class.
                Example: {'cat': 0, 'dog': 0} initially.
"""
                weights=1/(k_distances + 1e-9)
                class_weights={}
                for label,w in zip(k_labels,weights):
                    # .get(label, 0) → Gets the current total for that class, or 0 if it doesn’t exist yet.

                    # Adds the new neighbor’s weight to that total.
                    class_weights[label] = class_weights.get(label, 0) + w
                
                """
                s the classes by their total weight, highest first.

                .items() → Turns {'cat': 1.4, 'dog': 0.3} into [('cat', 1.4), ('dog', 0.3)]

                key=lambda x: x[1] → Sort by the weight (2nd element of tuple).

                reverse=True → Highest weight first."""
                sorted_classes=sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
            else:

                counts=Counter(k_labels)
                max_count=max(counts.values())

                tied_classes=[cls for cls , cnt in counts.items() if cnt == max_count]

                if len(tied_classes) > 1:
                    avg_dist = {cls: np.mean([d for lbl, d in zip(k_labels, k_distances) if lbl == cls]) 
                                for cls in tied_classes}
                    sorted_classes = sorted(avg_dist.items(), key=lambda x: x[1])
                    sorted_classes = [(cls, 1) for cls, _ in sorted_classes]
                else:
                   sorted_classes = [(tied_classes[0], 1.0)]
            
            if self.return_proba:
                total = sum(w for _, w in sorted_classes)
                return {cls: w / total for cls, w in sorted_classes}
            else:
                return sorted_classes[0][0]

        elif self.task == 'regression':
            if self.weighted:
                weights = 1 / (k_distances + 1e-9)
                return np.dot(weights, k_labels) / np.sum(weights)
            else:
                return np.mean(k_labels)
        else:
            raise ValueError("Task must be 'classification' or 'regression'")


class LineaRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        pass