import numpy as np


def best_split(X: np.ndarray, y: np.ndarray, columns: np.ndarray, min_samples_leaf: int) -> tuple[int, float, float] | None:
    """
    The column and threshold that minimise the squared error of the two groups, each predicted by its mean.

    For each column the houses are sorted once, and cumulative sums of y and y^2 give the error of every
    threshold at the same time, since the error of a group is sum(y^2) - sum(y)^2 / count.

    Parameters:
    X (np.ndarray): The columns of the houses in the node.
    y (np.ndarray): The target of the houses in the node.
    columns (np.ndarray): The columns to try.
    min_samples_leaf (int): The least number of houses allowed in a group.

    Returns:
    tuple or None: The column, the threshold and the squared error after the split, or None if no split is allowed.
    """
    n = len(y)
    left_count = np.arange(1, n)
    right_count = n - left_count
    best = None
    for j in columns:
        order = np.argsort(X[:, j], kind="stable")
        values, targets = X[order, j], y[order]
        total, total_square = targets.sum(), (targets**2).sum()
        left_sum = np.cumsum(targets)[:-1]
        left_square = np.cumsum(targets**2)[:-1]
        error = left_square - left_sum**2 / left_count + (total_square - left_square) - (total - left_sum) ** 2 / right_count
        allowed = (values[1:] != values[:-1]) & (left_count >= min_samples_leaf) & (right_count >= min_samples_leaf)
        if not allowed.any():
            continue
        error = np.where(allowed, error, np.inf)
        k = int(np.argmin(error))
        if best is None or error[k] < best[2]:
            best = (int(j), (values[k] + values[k + 1]) / 2, float(error[k]))
    return best


class RegressionTree:
    """A regression tree grown greedily by squared error. A leaf predicts the mean target of its houses."""

    def __init__(self, max_depth: int = None, min_samples_leaf: int = 1, max_features: int = None, seed: int = 0):
        """
        Parameters:
        max_depth (int): The most questions in a row. Default is no limit.
        min_samples_leaf (int): The least number of houses in a leaf.
        max_features (int): The number of random columns tried at each split. Default is all of them.
        seed (int): The seed of the random choice of columns.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTree":
        """Grow the tree, node by node, from the root."""
        rng = np.random.default_rng(self.seed)
        self.column, self.threshold, self.left, self.right, self.value = [], [], [], [], []

        def grow(rows: np.ndarray, depth: int) -> int:
            node = len(self.value)
            self.column.append(-1)
            self.threshold.append(0.0)
            self.left.append(-1)
            self.right.append(-1)
            self.value.append(y[rows].mean())
            if (self.max_depth is not None and depth >= self.max_depth) or len(rows) < 2 * self.min_samples_leaf or np.ptp(y[rows]) == 0:
                return node
            columns = np.arange(X.shape[1])
            if self.max_features is not None:
                columns = rng.choice(columns, size=self.max_features, replace=False)
            split = best_split(X[rows], y[rows], columns, self.min_samples_leaf)
            if split is None:
                return node
            j, threshold, _ = split
            goes_left = X[rows, j] <= threshold
            self.column[node], self.threshold[node] = j, threshold
            self.left[node] = grow(rows[goes_left], depth + 1)
            self.right[node] = grow(rows[~goes_left], depth + 1)
            return node

        grow(np.arange(len(y)), 0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Send the houses down the tree together, splitting the group at each question."""
        prediction = np.empty(len(X))
        stack = [(0, np.arange(len(X)))]
        while stack:
            node, rows = stack.pop()
            if self.left[node] == -1:
                prediction[rows] = self.value[node]
                continue
            goes_left = X[rows, self.column[node]] <= self.threshold[node]
            stack.append((self.left[node], rows[goes_left]))
            stack.append((self.right[node], rows[~goes_left]))
        return prediction


class RandomForest:
    """Bagged regression trees with random columns at each split. A prediction is the average of the trees."""

    def __init__(self, n_trees: int = 100, max_features: int = None, max_depth: int = None, min_samples_leaf: int = 1, seed: int = 0):
        """
        Parameters:
        n_trees (int): The number of trees.
        max_features (int): The number of random columns tried at each split. Default is all of them.
        max_depth (int): The most questions in a row. Default is no limit.
        min_samples_leaf (int): The least number of houses in a leaf.
        seed (int): The seed of the bootstrap samples and of the random columns.
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        """Grow each tree on a bootstrap sample, and keep which houses each tree did not see."""
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.trees, self.out_of_bag = [], []
        for _ in range(self.n_trees):
            sample = rng.integers(0, n, size=n)
            tree = RegressionTree(self.max_depth, self.min_samples_leaf, self.max_features, seed=int(rng.integers(2**31)))
            self.trees.append(tree.fit(X[sample], y[sample]))
            self.out_of_bag.append(np.setdiff1d(np.arange(n), sample))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The average prediction of the trees."""
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)

    def out_of_bag_prediction(self, X: np.ndarray) -> np.ndarray:
        """The prediction for each train house from only the trees that did not have it in their sample."""
        total, count = np.zeros(len(X)), np.zeros(len(X))
        for tree, rows in zip(self.trees, self.out_of_bag):
            total[rows] += tree.predict(X[rows])
            count[rows] += 1
        return total / count


class GradientBoosting:
    """Gradient boosting for squared error: each small tree is fitted to the residuals, and added with a learning rate."""

    def __init__(self, n_trees: int = 100, learning_rate: float = 0.1, max_depth: int = 3, min_samples_leaf: int = 1):
        """
        Parameters:
        n_trees (int): The number of trees.
        learning_rate (float): The share of each tree that is added, between 0 and 1.
        max_depth (int): The depth of each tree.
        min_samples_leaf (int): The least number of houses in a leaf.
        """
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        """Start from the mean, then repeatedly fit a tree to the residuals of the current prediction."""
        self.start = y.mean()
        prediction = np.full(len(y), self.start)
        self.trees = []
        for _ in range(self.n_trees):
            tree = RegressionTree(self.max_depth, self.min_samples_leaf).fit(X, y - prediction)
            prediction += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray, n_trees: int = None) -> np.ndarray:
        """The mean plus the learning rate times the trees, using the first n_trees if given."""
        prediction = np.full(len(X), self.start)
        for tree in self.trees[:n_trees]:
            prediction += self.learning_rate * tree.predict(X)
        return prediction
