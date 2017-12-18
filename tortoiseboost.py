from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF
from sklearn.metrics import mean_absolute_error
import cplex


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


class LP_model(object):
    def __init__(self, y, reg_alpha=1.0, n_estimators=10, max_leaf_nodes=3):
        """Initialize the cplex model used in the TortoiseBoostRegressor.
        LAD Lasso Regression is used in the fully corrective step and is 
        implemented as a linear programming problem.."""
        model = cplex.Cplex()
        # Set display parameters to avoid output being printed to the terminal.
        model.parameters.barrier.display.set(0)
        model.parameters.simplex.display.set(0)

        # Minimize the objective function
        model.objective.set_sense(model.objective.sense.minimize)
        n = y.shape[0]
        J = max_leaf_nodes
        K = n_estimators

        # Create epsilon constraints
        rhs = list(y)*2
        senses = 'L'*n + 'G'*n
        model.linear_constraints.add(rhs=rhs,
                                     senses=senses)
        c_epsilon_upper_indices = range(0, n)
        c_epsilon_lower_indices = range(n, 2*n)
        
        # Create delta constraints
        rhs = [0.0]*2*J*K
        senses = 'L'*J*K + 'G'*J*K
        model.linear_constraints.add(rhs=rhs,
                                     senses=senses)
        c_delta_upper_indices = range(2*n, 2*n + J*K)
        c_delta_lower_indices = range(2*n + J*K, 2*n + 2*J*K)

        # Create epsilon variables
        objective = [1.0]*n
        lower_bounds = [0.0]*n
        upper_bounds = [1e20]*n
        columns = [cplex.SparsePair(ind=[c_epsilon_upper_indices[i],
                                         c_epsilon_lower_indices[i]],
                                    val=[-1.0, 1.0]) for i in range(n)]
        model.variables.add(obj=objective,
                            columns=columns,
                            ub=upper_bounds,
                            lb=lower_bounds)
        epsilon_indices = range(n)

        # Create delta variables
        objective = [reg_alpha]*K*J
        lower_bounds = [0.0]*K*J
        upper_bounds = [1e20]*K*J
        columns = [cplex.SparsePair(ind=[c_delta_upper_indices[i],
                                         c_delta_lower_indices[i]],
                                    val=[-1.0, 1.0]) for i in range(K*J)]
        model.variables.add(obj=objective,
                            columns=columns,
                            ub=upper_bounds,
                            lb=lower_bounds)
        delta_indices = range(n, n + J*K)

        # Create an intercept variable
        objective = [0.0]
        lower_bound = [-1e20]
        column = [cplex.SparsePair(ind=range(0, 2*n), val=[1.0]*2*n)]
        model.variables.add(obj=objective,
                            columns=column,
                            lb=lower_bound)

        self.model = model
        self.y = y
        self.reg_alpha = reg_alpha
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.c_epsilon_upper_indices = c_epsilon_upper_indices
        self.c_epsilon_lower_indices = c_epsilon_lower_indices
        self.c_delta_upper_indices = c_delta_upper_indices
        self.c_delta_lower_indices = c_delta_lower_indices
        self.epsilon_indices = epsilon_indices
        self.delta_indices = delta_indices
        self.intercept_index = n+J*K
        self.weight_indices = []
        self.leaves_per_tree = []
        self.iteration = 0

    def update(self, terminal_regions, leaves):
        """Updates the lp model. Regression problem has a new predictor
        for each leaf in the new tree. For each data point x, the predictor
        j has value 1 if x falls into leaf j of the new tree. It has value 0
        otherwise."""
        n = self.y.shape[0]
        J = self.max_leaf_nodes
        L = len(leaves)

        num_previous_leaves = sum(self.leaves_per_tree)
        k = num_previous_leaves
        K = self.n_estimators
        
        objective = [0.0]*L
        lower_bounds = [-1e20]*L
        columns = [cplex.SparsePair(ind=self.c_epsilon_upper_indices +
                                    self.c_epsilon_lower_indices +
                                    [2*n+k+l,
                                     2*n+J*K+k+l],
                                    val=[1.0 if terminal_region == leaves[l]
                                         else 0.0
                                         for terminal_region
                                         in terminal_regions]*2
                                    + [1.0, 1.0])
                   for l in range(L)]
        self.weight_indices.append(range(n + J*K + num_previous_leaves + 1,
                                         n + J*K + num_previous_leaves + L + 1))
        self.model.variables.add(obj=objective,
                                 lb=lower_bounds,
                                 columns=columns)

        self.model.solve()

        self.leaves_per_tree.append(len(leaves))

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
        self.X_ = X
        self.y_ = y
        n, p = self.X_.shape

        # Initialize solver for LAD Lasso. We build the constraint matrix
        # incrementally.
        lp_solver = LP_model(y, reg_alpha=self.reg_alpha,
                             n_estimators=self.n_estimators,
                             max_leaf_nodes=self.max_leaf_nodes)

        models = []
        leaves_list = []
        terminal_regions_list = []
        # Initial estimate given by median of response
        self.h0 = np.median(y)

        for iteration in range(self.n_estimators):
            residuals = np.sign(y - self.h0 - model_sum(models, X))
            # Fit a new decision tree to the psuedoresiduals
            base_model = DecisionTreeRegressor(
                criterion='mse',
                splitter='best',
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=None,
                presort=True)
            base_model.fit(X, residuals)

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

