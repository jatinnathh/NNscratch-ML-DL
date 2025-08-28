import numpy as np
import pandas as pd
from collections import Counter
from typing import Callable, Union
from .stats import euclidean_distance as euc_dist 
from .stats import manhattan_distance 



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
        X=np.array(X)
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


class LinearRegression:
    def __init__(self, fit_intercept=True, method="normal",lr=0.01, n_iterations=1000):
        """
        fit_intercept: whether to add a bias term
        method: 'normal' for normal equation, 'gd' for gradient descent
        lr: learning rate (used if method='gd')
        n_iters: number of iterations (used if method='gd')
        """

        self.fit_intercept = fit_intercept
        self.method = method
        self.lr = lr
        self.n_iterations = n_iterations

        self.coef_ = None
        self.intercept_ = None

    
    def add_bias(self,X):
        # adds a columns of 1s for bias
        if self.fit_intercept:
            bias = np.ones((X.shape[0], 1))
            X = np.hstack((bias, X))
        return X
    

    def fit(self,X,y):
        X=np.array(X,dtype=float)
        y=np.array(y,dtype=float).reshape(-1,1)
        X_b=self.add_bias(X)

        if self.method=="normal":
            #  normal equation
            theta_best=np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        elif self.method=="gd":
            m,n=X_b.shape
            theta=np.zeros((n,1))

            """
            X_b.dot(theta) → predicted values (ŷ) for current theta.

           ( X_b.dot(theta) - y) → error vector (ŷ - y).

            X_b.T.dot(error) → sum of error contributions for each feature.

            (2/m) → scaling factor from the derivative of the MSE loss.

            Result: gradients is a column vector of partial derivatives, one per parameter.
            """
            for _ in range(self.n_iterations):
                gradients=(2/m)*X_b.T.dot(X_b.dot(theta) - y)
                theta-=self.lr*gradients
            theta_best=theta
        else:
            raise ValueError("Method must be 'normal' or 'gd'.")

        # Store coefficients
        if self.fit_intercept:
            self.intercept_ = theta_best[0, 0]
            self.coef_ = theta_best[1:, 0]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta_best[:, 0]

    def predict(self,X):
        X=np.array(X,dtype=float)
        X_b=self.add_bias(X)
        if self.fit_intercept:
            #  Combine bias & weights into one vector
            theta=np.r_[self.intercept_, self.coef_]
        else: theta=self.coef_

        return X_b.dot(theta)
    
    def score(self,X,y):
        # R2 
        y=np.array(y,dtype=float).reshape(-1,1)
        y_pred=self.predict(X)

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_total) if ss_total > 0 else 0
    


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, fit_intercept=True):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.weights = None
        self.intercept_ = None

    def _add_bias(self, X):
        if self.fit_intercept:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)  # Prevent overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        X_b = self._add_bias(X)

        m, n = X_b.shape
        self.weights = np.zeros((n, 1))

        for _ in range(self.n_iters):
            linear_model = X_b.dot(self.weights)
            y_pred = self._sigmoid(linear_model)
            gradients = (1/m) * X_b.T.dot(y_pred - y)
            self.weights -= self.lr * gradients

        if self.fit_intercept:
            self.intercept_ = self.weights[0, 0]
            self.coef_ = self.weights[1:, 0]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.weights[:, 0]

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        X_b = self._add_bias(X)
        return self._sigmoid(X_b.dot(self.weights)).flatten()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == np.array(y))


class DecisionTreeClassifier:
    def __init__(self,max_depth=None,min_samples_split=2):
        """
        max_depth: maximum depth of the tree (None -> grow until pure or min_samples_split reached)
        min_samples_split: minimum samples needed to split a node
        """
         
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    
    def _gini(self,y):
        """
        A measure of how pure a node in a decision tree is.

        Interpretation:

        0 → perfectly pure (all samples belong to the same class).

        Higher values → more mixed classes.
        
        """

        m=len(y)
        if m==0:
            return 0
        #  np.unique(y, return_counts=True) → finds unique classes in y and how many samples belong to each.
        _,counts = np.unique(y, return_counts=True)
        probs=counts / m
        return 1-np.sum(probs ** 2)
    
    def _best_split(self,X,y):
        m,n=X.shape
        if m < self.min_samples_split:
            return None,None
        """
        Start with the worst possible impurity (1).

        split_idx = which feature to split on.

        split_threshold = the value of that feature to split at.    """
        best_gini = 1
        split_idx, split_threshold = None, None

        # For each feature (column), get all unique values(threshold) — these are the only meaningful split points.
        for feature_idx in range(n):
            thresholds=np.unique(X[:, feature_idx])
            for threshold in thresholds:
                """Create a boolean mask left_mask where samples have the feature value less than or equal to the threshold.

                right_mask is the opposite mask (samples with feature value greater than threshold)."""
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # If all samples fall into only one side of the split (either left or right group is empty), 
                # skip this threshold because it's not a valid split.
                if left_mask.sum()==0 or right_mask.sum()==0:
                    continue
                gini_left=self._gini(y[left_mask])
                gini_right=self._gini(y[right_mask])

                # Calculate the weighted average Gini impurity after the split.

                # Weight the Gini impurity of each side by the proportion of samples falling into that side.
                weighted_gini = (left_mask.sum() * gini_left + right_mask.sum() * gini_right) / m
                
                """
                if this split produces a lower weighted Gini impurity than the current best:

                Update the best Gini impurity value.

                Store the current feature index and threshold as the best split found so far.
                """
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    split_idx = feature_idx
                    split_threshold = threshold
        return split_idx, split_threshold
    
    def _build_tree(self,X,y,depth=0):
        num_samples,num_features=X.shape
        num_labels=len(np.unique(y))

        if (self.max_depth is not None and depth>=self.max_depth or num_labels==1 or num_samples < self.min_samples_split):
            #  If max depth reached, pure node, or not enough samples to split, return the most common label.
            leaf_value=self._most_common_label(y)
            return {"leaf": leaf_value}
        

        feature_idx,threshold=self._best_split(X,y)
        
        # If no split is found (no improvement possible), make this node a leaf with the most common label.
        if feature_idx is None:
            leaf_value=self._most_common_label(y)
            return {"leaf": leaf_value}
        
        """
        Create masks to split the dataset into two subsets:

        Left subset: samples where the chosen feature's value is less than or equal to the threshold.

        Right subset: samples where the feature's value is greater than the threshold."""
        left_mask=X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Recursively build the left subtree and right subtree by calling _build_tree on the respective subsets.

        # Increase the depth counter by 1 for these recursive calls. 
        left_subtree=self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree=self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }
        # Return a dictionary representing the current decision node:

        # Which feature is used for splitting (feature)
        # The threshold value
        # The left subtree (for samples <= threshold)
        # The right subtree (for samples > threshold)

    def _most_common_label(self, y):
        """Return the most common label in y"""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self,X,y):
        X=np.array(X,dtype=float)
        y=np.array(y)
        self.tree=self._build_tree(X,y)

    
    def _predict_one(self,inputs,node):
        if "leaf" in node:
            return node["leaf"]
        feature_val = inputs[node["feature"]]
        if feature_val <= node["threshold"]:
            return self._predict_one(inputs, node["left"])
        else:
            return self._predict_one(inputs, node["right"])
    
    def predict(self, X):
        X = np.array(X, dtype=float)
        return [self._predict_one(sample, self.tree) for sample in X]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == np.array(y))





class RandomForestClassifier:
    def __init__(self,n_estimators=100,max_depth=None,min_samples_split=2,
                 max_features="sqrt",bootstrap=True,random_state=None):
        
        """
        n_estimators: number of trees
        max_depth: max depth of each tree
        min_samples_split: minimum samples to split a node
        max_features: number of features to consider when looking for the best split
                      ('sqrt' => sqrt of total features, 'log2' => log2(total features), int or float)
        bootstrap: whether to use bootstrap samples (default True)
        random_state: for reproducibility
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []
        self.feature_indices = [] 

        if random_state is not None:
            np.random.seed(random_state)

    def _get_max_features(self, n_features):
        # This function decides how many of them should be used when looking for the best split at a tree node.
        if isinstance(self.max_features, int):
            # If the user explicitly says max_features=5, it just returns 5
            return self.max_features
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            # Default to all features
            return n_features


    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        n_samples,n_features = X.shape

        max_feats=self._get_max_features(n_features)

        self.trees=[]
        self.feature_indices=[]

        for _ in range(self.n_estimators):
            """
            if self.bootstrap:
            self.bootstrap is a boolean flag.

            If True, the tree will be trained on a bootstrap sample of the dataset.

            If False, it will just use the full dataset without resampling.

                        """
        

            """
            Bootstrapping increases randomness in each tree → reduces overfitting when combining multiple trees.

            In Random Forests, each tree sees a slightly different version of the dataset, making the ensemble more robust."""
            if self.bootstrap:

                """
                n_samples = total number of rows in the training data.

                np.random.choice(..., replace=True) → randomly selects rows with replacement.

                This means the same row can appear multiple times in the sample, and some rows from the original dataset might not appear at all.

                This creates a dataset of the same size but slightly different distribution."""
                indices=np.random.choice(n_samples,size=n_samples,replace=True)
                X_sample=X[indices]
                y_sample=y[indices]
            else:
                X_sample=X
                y_sample=y

            # This part is doing random column (feature) selection for each tree in your Random Forest.
            """
            n_features-> total number of column sin X
            max_feats-> number of faetures rto condsider for this tree
            The result (feat_indices) is an array of column indices.


            """
            feature_indices=np.random.choice(n_features,size=max_feats,replace=False)

            self.feature_indices.append(feature_indices)

            tree=DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:,feature_indices],y_sample)
            self.trees.append(tree)
    

    def predict(self,X):
        X=np.array(X)


        """
        Creates an empty 2D array to store predictions from each tree.

        Shape:

        Rows = number of samples in X

        Columns = number of trees (self.n_estimators)


        """
        tree_preds=np.zeros((X.shape[0],self.n_estimators),dtype=object)


        """
        feat_idx → retrieves the column indices that this tree was trained on.

        X[:, feat_idx] → selects only those columns for prediction.

        tree.predict(...) → gets predictions from that tree.

        Saves them in tree_preds.
        """
        for i,tree in enumerate(self.trees):
            feat_idx=self.feature_indices[i]
            preds=tree.predict(X[:,feat_idx])
            tree_preds[:,i]=preds
        
        y_pred=[]




        """
        For each row (sample) in tree_preds:

        Count how many trees predicted each label.

        Take the label with the highest count → majority vote.

        y_pred ends up as the final prediction for all samples."""
        for pred in tree_preds:
            counts=Counter(pred)
            y_pred.append(counts.most_common(1)[0][0])
        
        return y_pred
    
    def score(self, X, y):
        y_pred=self.predict(X)
        return np.mean(y_pred == np.array(y))




class LinearSVM:
    """
    Linear Support Vector Machine (SVM) Classifier from scratch.
    - Uses Hinge Loss + L2 Regularization
    - Trained with Gradient Descent (subgradient method)
    Works like sklearn's LinearSVC (for binary classification).
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Parameters:
        - learning_rate : float, step size for gradient descent
        - lambda_param  : float, regularization strength (controls margin width)
        - n_iters       : int, number of training iterations
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None   # weight vector
        self.b = None   # bias term

    def fit(self, X, y):
        """
        Train the SVM using gradient descent on hinge loss.
        X : (n_samples, n_features) feature matrix
        y : (n_samples,) labels, must be in {-1, 1}
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Convert labels to {-1, 1} if needed (SVM requires this form)
        y_ = np.where(y <= 0, -1, 1)

        # Gradient Descent Loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condition: y_i * (w·x + b) >= 1  (correct classification with margin)
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b)
                  # Hinge Loss condition:
                # If yi(w·xi + b) >= 1 -> correctly classified outside margin -> no loss
                # If yi(w·xi + b) < 1 -> inside margin or misclassified -> update needed
                if condition >= 1:
                    # Correct classification, inside margin → only apply regularization
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # Misclassified or within margin
                    dw = 2 * self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]

                # Update weights & bias
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        """
        Predict class labels {-1, 1} for input samples.
        """
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

    def score(self, X, y):
        """
        Return accuracy of the model.
        """
        preds = self.predict(X)
        y_true = np.where(y <= 0, -1, 1)
        return np.mean(preds == y_true)

# about hinge loss
# Hinge loss is used for "maximum-margin" classification, most notably for support vector machines.
# The goal of SVM is to maximize the margin between classes 
# while penalizing misclassifications.
#
# For one training sample (xi, yi):
#   L_i = max(0, 1 - yi(w·xi + b))
#
# Cases:
# 1) yi(w·xi + b) >= 1:
#       → Correctly classified and outside margin → loss = 0
#
# 2) 0 < yi(w·xi + b) < 1:
#       → Correctly classified but inside margin → small loss > 0
#
# 3) yi(w·xi + b) < 0:
#       → Misclassified → large loss
#
# Total Loss with Regularization:
#   L = (1/n) * Σ max(0, 1 - yi(w·xi + b)) + λ||w||²
#
# First term → encourages classification with margin
# Second term → prevents weights from growing too large (regularization)





class SVC:
    def __init__(self,C=1,lr=0.001,kernel="rbf",gamma=0.5,tol=1e-3,max_passess=3):
        self.C = C                   # Regularization strength (upper bound for alphas)
        self.lr = lr                 # Learning rate
        self.kernel = kernel         # "linear" or "rbf"
        self.gamma = gamma           # RBF kernel parameter
        self.tol = tol               # Numerical tolerance for KKT checks
        self.max_passess = max_passess  # Maximum number of passes over the training data

    def _kernel(self,x1,x2):
        # Define kernel function   
        if self.kernel == "linear":
            return np.dot(x1, x2)
        elif self.kernel == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unknown kernel")


    def fit(self,X,y):
        # Convert labels to {-1, +1}
        self.X=X
        self.y=np.where(y==0,-1,1)
        n_samples,n_features=X.shape

        # Initialize dual variables (alphas) and bias
        self.alpha=np.zeros(n_samples)
        self.b=0
        passes=0


        # Precompute Gram (kernel) matrix K[i,j] = k(x_i, x_j)
        k=np.zeros((n_samples,n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                k[i,j]=self._kernel(X[i],X[j])

        while passes < self.max_passess:
            num_changed_alpha=0
            for i in range(n_samples):

                #  compute error for sample i
                Ei=self._decision_function(X[i]) - self.y[i]
                
                #  check KKT conditions
                if (self.y[i]*Ei<-self.tol and self.alpha[i]<self.C )or self.y[i]*Ei>self.tol and self.alpha[i]>0:
                    # update 2 alphas at a time to maintain equality 
                    j=np.random.randint(0,n_samples-1)
                    if j==i:
                        j=(j+1)%n_samples
                    Ej=self._decision_function(X[j]) - self.y[j]
                    alpha_i_old,alpha_j_old=self.alpha[i],self.alpha[j]

                    
                    # When updating alpha[i] and alpha[j], 
                    # we need to keep the constraint:  0 <= alpha <= C 
                    # and also:   sum(y[i] * alpha[i]) = 0
                    # So we compute bounds L and H between which alpha[j] must lie.

                    if self.y[i] != self.y[j]:
                        # Case 1: labels are different
                        # Lower bound (L): alpha[j] - alpha[i] but not below 0
                        L = max(0, self.alpha[j] - self.alpha[i])
                        # Upper bound (H): C + alpha[j] - alpha[i], but not above C
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        # Case 2: labels are same
                        # Lower bound (L): alpha[i] + alpha[j] - C, but not below 0
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        # Upper bound (H): alpha[i] + alpha[j], but not above C
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    # After this, alpha[j] must stay within [L, H].
                    # If L == H, no update is possible for alpha[j].
                    
                    if L==H:
                        continue

                    eta=2*k[i,j]-k[i,i]-k[j,j]
                    if eta >=0:
                        continue

                    self.alpha[j]-=self.y[j]*(Ei - Ej)/eta
                    self.alpha=np.clip(self.alpha[j],L,H)
                    if abs(self.alpha[j]-alpha_j_old)<1e-5:
                        continue


                    self.alpha[i]+=self.y[i]*self.y[j]*(alpha_j_old - self.alpha[j])


                    b1=self.b-Ei -  self.y[i]*(self.alpha[i]-alpha_i_old)*k[i,i] - self.y[j]*(self.alpha[j]-alpha_j_old)*k[i,j]
                    b2=self.b-Ej -  self.y[i]*(self.alpha[i]-alpha_i_old)*k[i,j] - self.y[j]*(self.alpha[j]-alpha_j_old)*k[j,j]
                    

                    if 0<self.alpha[i]<self.C:
                        self.b=b1
                    elif 0<self.alpha[j]<self.C:
                        self.b=b2
                    else:
                        self.b=(b1+b2)/2
                    num_changed_alpha+=1


                if num_changed_alpha==0:
                    passes+=1
                else:
                    passes=0
    def _decision_function(self,X):
        res=0

        for i in range(len(self.alpha)):
            if self.alpha>0:
                res+=self.alpha[i]*self.y[i]*self._kernel(X,self.X[i])
        
        return res


    def predict(self,X):
        preds=[]

        for x in X:
            pred=np.sign(self._decision_function(x))
            preds.append(1 if pred==1 else 0)
        return np.array(preds)
    