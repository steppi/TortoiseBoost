from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF
from sklearn.metrics import mean_absolute_error
from scipy import sparse


# Utility functions for the TortoiseBoostRegressor
def model_sum(models, X):
    """Given a list of models and predictors X, return the sum of the
    predictions from each model."""
    n, p = X.shape
    preds = np.zeros(n)
    for model in models:
        preds += np.fromiter((pred for pred in model.predict(X)),
                             dtype=np.float64)
    return preds


def tree_compress(tree, X):
    """Given a sklearn tree object and a dataset X,
    return an array that contains that indices of the leaves in that
    stores the nodes of the tree. Also return an array storing which leaf each
    point of X maps into."""
    # Must convert X to np.float32 for use in sklearn.tree Tree data structure
    terminal_regions = tree.apply(np.array(X, dtype=np.float32))
    leaves = list(np.where(tree.children_left == TREE_LEAF)[0])
    return (terminal_regions, leaves)


def change_weights(model, leaves, new_values):
    """Change the weights of a tree."""
    assert(len(leaves) == len(new_values))
    tree = model.tree_
    for index, leaf in enumerate(leaves):
        tree.value[leaf, 0, 0] = new_values[index]
    return


class Solver(object):
    def __init__(self, y, reg_alpha=1.0, n_estimators=10, max_leaf_nodes=3):
        """Coordinate descent solver for LAD Lasso Regression"""
        n = y.shape[0]
        J = max_leaf_nodes
        K = n_estimators

        design = sparse.hstack(sparse.lil_matrix(K*J, n),
                               self.reg_alpha * sparse.eye(K*J, format='lil'))
        design = sparse.vstack([np.hstack(np.full(n, 1),
                                          np.zeros(K*J)),
                                design])
        initial = np.median(y)
        residuals = np.hstack([y, np.zeros(K*J)]) - initial
        dd = np.array([-np.sum(np.sign(residuals)),
                       np.sign(residuals)])
        dd = np.vstack([dd, np.full((K*J, 2), reg_alpha)])
        parameters = np.zeros(J*K+1)
        parameters[0] = initial
        parameters.shape = (J*K+1, 1)

        self.design = design
        self.residuals = residuals
        self.dd = dd
        self.parameters = parameters
        self._num_previous_leaves = 0
        self._J = J
        self._K = K
        self._n = n

    def update(self, terminal_regions, leaves):
        """Updates the solver. Regression problem has a new predictor
        for each leaf in the new tree. For each data point x, the predictor
        j has value 1 if x falls into leaf j of the new tree. It has value 0
        otherwise."""
        n = self.y.shape[0]
        L = len(leaves)

        num_previous_leaves = np.sum(self._leaves_per_tree)
        k = num_previous_leaves
        K = self.n_estimators

        self.design[k:k+L, :n] = ([1. if terminal_region == leaves[l]
                                   else 0.
                                   for terminal_region in terminal_regions]
                                  for l in range(L))

        self.dd[k:K+L, :] = self.design[k:k+L, :].dot(np.sign(self.residuals))
        self._num_previous_leaves += L

    def solve(self):
        """Find solution"""
        min_dd = np.unravel_index(np.argmin(self.dd), (self._K*self._J+1, 2))
        while self.dd[min_dd] < 0:
            b = min_dd[0]
            min_dd = np.unravel_index(np.argmin(self.dd),
                                      (self._K*self._J+1, 2))
            # find a better variable name
            g = np.zeros(self._n+1)
            g0 = -self.parameters[0]
            g1 = self._design[b, :self._n].dot(self.residuals[:self._n])
            g1 = np.fromiter((v for v, i in enumerate(g1)
                              if self.design[b, i] != 0), dtype=np.float)
            g1 = g1 + self._parameters[b]
            g = np.hstack([g0, g1])
            weights = np.full((self._n+1, 1), 1)
            weights[0] = self.reg_alpha
            g = np.vstack([g, weights])
            g = g[:, g[0, :].argsort()]

    def get_weights(self):
        """Get the tree weights from solution of linear programming problem."""
        values = self.model.solution.get_values()
        weights = [[values[i] for i in s]
                   for s in self.weight_indices]
        intercept = values[self.intercept_index]
        return (weights, intercept)


class TortoiseBoostRegressor(BaseEstimator, RegressorMixin):
    """LAD_TreeBoost with full correction. Full correction is done using
    LAD Lasso regression."""
    def __init__(self, reg_alpha=1.0, n_estimators=10, max_leaf_nodes=5):
        self.n_estimators = n_estimators
        self.reg_alpha = reg_alpha
        self.max_leaf_nodes = max_leaf_nodes
        self.models_ = []

    def fit(self, X, y=None):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse=True)
        n, p = X.shape

        models = []
        leaves_list = []
        terminal_regions_list = []
        # Initial estimate given by median of response
        self.h0 = np.median(y)
        K, J = len(y), self.max_leaf_nodes
        

        for iteration in range(self.n_estimators):
            # Fit a new decision tree to the psuedoresiduals
            base_model = DecisionTreeRegressor(
                criterion='mse',
                splitter='best',
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=None,
                presort=True)
            base_model.fit(X, np.sign(residuals))

            tree = base_model.tree_
            # Extract the indices of the leaves of the tree and a list of which
            # of the leaves each datapoint X falls into
            terminal_regions, leaves = tree_compress(tree, X)
            # If the newly generated tree contains only one leaf, we do not
            # use it and skip to the next iteration.
            if len(leaves) == 1:
                continue

            # Update the lp model with the new variables and constraints
            # lp_solver.update(terminal_regions, leaves, self.reg_alpha)
            lp_solver.update(terminal_regions, leaves)

            # Add the tree to our list of models
            models.append(base_model)

            leaves_list.append(leaves)
            terminal_regions_list.append(terminal_regions)

            # Update the weights of each tree based on the solution to the
            # LAD Lasso problem
            all_weights, intercept = lp_solver.get_weights()
            for index, info in enumerate(zip(leaves_list,
                                             all_weights)):
                model = models[index]
                change_weights(model, info[0], info[1])
            self.h0 = intercept

        self.models_ = models
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, "models_")
        # Input validation
        X = check_array(X, accept_sparse=True)
        return self.h0 + model_sum(self.models_, X)

    def score(self, X, y):
        return mean_absolute_error(self.predict(X), y)

