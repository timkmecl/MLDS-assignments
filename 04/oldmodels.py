# LogReg


import numpy as np
from scipy.optimize import fmin_l_bfgs_b




def log_score(y, p):
    samples_range = np.arange(len(y))
    return np.mean(np.log(p[samples_range, y]))




class MultinomialLogReg:
    def __init__(self):
        pass
    
    def softmax(u):
        u = np.hstack((u, np.zeros((u.shape[0], 1))))
        u = u - np.max(u, axis=1, keepdims=True)
        exps = np.exp(u)
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def build(self, X, y, intercept=True):
        classes_count = int(np.max(y) + 1)
        feats_count = X.shape[1]

        if intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            feats_count += 1
        
        samples_range = np.arange(X.shape[0])
        def loss(betas):
            betas = betas.reshape((feats_count, classes_count-1))
            probs = MultinomialLogReg.softmax(X @ betas)
            return -np.sum(np.log(probs[samples_range, y]))

        betas_0 = np.zeros((feats_count, classes_count-1))
        betas_opt, _, _ = fmin_l_bfgs_b(loss, betas_0.flatten(), approx_grad=True, maxiter=10000)
        betas_opt = betas_opt.reshape((feats_count, classes_count-1))

        return MultinomialModel(classes_count, feats_count, betas_opt, intercept)


class MultinomialModel:
    def __init__(self, classes_count, feats_count, betas, intercept):
        self.classes_count = classes_count
        self.feats_count = feats_count
        self.betas = betas
        self.intercept = intercept
    
    def predict(self, X, epsilon=0):
        if self.intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        probs = MultinomialLogReg.softmax(X @ self.betas)
        probs = epsilon + (1 - 2 * epsilon) * probs
        return probs
    






# Tree and Forest

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt




def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    c = np.arange(X.shape[1])
    rand.shuffle(c)
    c = c[:int(np.sqrt(X.shape[1]))]
    return c




class Tree:
    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples
        
    def gini(self, y):
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)
    
    def gain(self, X, y, column, thr, current_gini):
        # gain of a given split
        under_thr = X[:, column] < thr
        y_under = y[under_thr]
        y_over = y[~under_thr]
        new_gini = (len(y_under) * self.gini(y_under) + len(y_over) * self.gini(y_over)) / len(y)

        return current_gini - new_gini

    def build(self, X, y):
        # if there is too few samples or all the same return a leaf with majority class
        if len(y) < self.min_samples or len(y) == 1 or len(np.unique(y)) == 1:
            return TreeNode(np.bincount(y).argmax())
        
        # find the best split
        current_gini = self.gini(y)
        best_gain, best_column, best_thr = 0, 0, 0
        columns = self.get_candidate_columns(X, self.rand)
        for column in columns:
            # go over all possible splits
            sorted_ids = np.argsort(X[:, column])
            for i in range(1, len(X)):
                if X[sorted_ids[i], column] != X[sorted_ids[i-1], column]:
                    thr = (X[sorted_ids[i], column] + X[sorted_ids[i-1], column]) / 2
                    gain = self.gain(X, y, column, thr, current_gini)
                    if gain > best_gain:
                        best_gain, best_column, best_thr = gain, column, thr
        
        # if no improvement by splitting return a leaf
        if best_gain == 0:
            return TreeNode(np.bincount(y).argmax())
        
        # build the subtrees using best split
        under_thr = X[:, best_column] < best_thr
        X_under, y_under = X[under_thr], y[under_thr]
        X_over, y_over = X[~under_thr], y[~under_thr]
        left = self.build(X_under, y_under)
        right = self.build(X_over, y_over)

        return TreeNode(None, left, right, best_column, best_thr)



class TreeNode:
    def __init__(self, prediction, left=None, right=None, column=None, thr=None):
        self.prediction = prediction
        self.left = left
        self.right = right
        self.column = column
        self.thr = thr

    def predict(self, X):
        # if it is a leaf return the prediction
        if self.prediction is not None:
            return self.prediction
        
        # otherwise split the data and recursively predict
        under_thr = X[:, self.column] < self.thr
        pred_left = self.left.predict(X[under_thr])
        pred_right = self.right.predict(X[~under_thr])

        # merge predictions
        pred = np.empty(len(X), dtype=int)
        pred[under_thr] = pred_left
        pred[~under_thr] = pred_right

        return pred
    




class RandomForest:
    def __init__(self, rand=None, n=50):
        self.n = n
        self.rand = rand
        self.rftree = Tree(rand=rand,
                            get_candidate_columns=random_sqrt_columns,
                            min_samples=1)
    
    def build(self, X, y):
        bags = []
        trees = []
        oob = []

        # bootstrap samples
        for i in range(self.n):
            ids = self.rand.choices(range(X.shape[0]), k=len(X))
            bags.append(ids)
            oob_i = [i for i in range(len(X)) if i not in ids]
            oob.append(oob_i)
            X_bag, y_bag = X[ids], y[ids]
            trees.append(self.rftree.build(X_bag, y_bag))
        
        return RFModel(self.n, bags, trees, oob, self.rand, X, y)



class RFModel:
    def __init__(self, n, bags, trees, oob, rand, X, y):
        self.n = n
        self.bags = bags
        self.trees = trees
        self.oob = oob
        self.rand = rand
        self.X = X
        self.y = y

    def predict(self, X):
        pred = np.zeros(len(X), dtype=int)
        for tree in self.trees:
            pred += tree.predict(X)
        
        pred = np.round(pred / self.n)
        return pred

    def importance(self):
        def misscl_rate(y, preds):
            errors = preds != y
            return errors.sum() / len(y)
        
        n_features = self.X.shape[1]
        imps = np.zeros(n_features)
        # go over all features
        for feat in range(n_features):
            # go over all trees
            for i in range(self.n):
                oob_ids = self.oob[i]
                oob_X = self.X[oob_ids, :]
                oob_y = self.y[oob_ids]
                miss_before = misscl_rate(oob_y, self.trees[i].predict(oob_X))

                # permute the feature
                permuted_X = oob_X.copy()
                self.rand.shuffle(permuted_X[:, feat])
                miss_after = misscl_rate(oob_y, self.trees[i].predict(permuted_X))
                diff = miss_after - miss_before
                imps[feat] += diff

        return imps / self.n