import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_indices]
    target_vector = target_vector[sorted_indices]

    unique_values = np.unique(feature_vector)
    if len(unique_values) <= 1:
        return np.array([]), np.array([]), None, None

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    ginis = []

    total_samples = len(target_vector)

    for threshold in thresholds:
        left_mask = feature_vector < threshold
        right_mask = ~left_mask

        y_left = target_vector[left_mask]
        y_right = target_vector[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            ginis.append(-np.inf)
            continue

        def gini(y):
            p1 = np.mean(y == 1)
            p0 = np.mean(y == 0)
            return 1 - p1 ** 2 - p0 ** 2

        H_left = gini(y_left)
        H_right = gini(y_right)

        Q = -(len(y_left) / total_samples) * H_left - (len(y_right) / total_samples) * H_right
        ginis.append(Q)

    ginis = np.array(ginis)
    valid_thresholds = thresholds[ginis != -np.inf]
    valid_ginis = ginis[ginis != -np.inf]

    if len(valid_ginis) == 0:
        return thresholds, ginis, None, None

    best_idx = np.argmax(valid_ginis)
    threshold_best = valid_thresholds[best_idx]
    gini_best = valid_ginis[best_idx]

    return valid_thresholds, valid_ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("Неизвестный тип признака")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = int(sub_y[0]) 
            return

        if (
            self._max_depth is not None and depth >= self._max_depth or
            len(sub_X) < self._min_samples_split
        ):
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        categories_map_best = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: clicks.get(key, 0) / count for key, count in counts.items()}
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {
                    category: i for i, category in enumerate(sorted_categories)
                }
                feature_vector = np.vectorize(categories_map.get)(sub_X[:, feature])
            else:
                raise ValueError("Некорректный тип признака")

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is None:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini

                if feature_type == "real":
                    threshold_best = threshold
                    split = feature_vector < threshold
                    categories_map_best = None
                elif feature_type == "categorical":
                    threshold_best = threshold
                    categories_map_best = categories_map
                    split = np.isin(feature_vector, np.arange(threshold))

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = float(threshold_best)
        elif self._feature_types[feature_best] == "categorical":
            left_categories = [k for k, v in categories_map_best.items() if v < threshold_best]
            node["categories_split"] = left_categories

        node["left_child"], node["right_child"] = {}, {}
        try:
            self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
            self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)
        except Exception as e:
            print(f"Ошибка при обучении поддерева на признаке {feature_best}: {e}")
    
    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание для одного объекта.
        """
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            threshold = node.get("threshold")
            if threshold is None:
                raise KeyError("Threshold отсутствует в узле")
            if float(x[feature_idx]) < threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif feature_type == "categorical":
            category = x[feature_idx]
            left_categories = node.get("categories_split")
            if left_categories is None:
                raise KeyError("categories_split отсутствует в узле")
            if category in left_categories:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        else:
            raise ValueError("Некорректный тип признака")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

class CustomDecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=2):
        self.tree = DecisionTree(feature_types=feature_types,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split)

    def fit(self, X, y):
        self.tree.fit(X, y)
        return self

    def predict(self, X):
        return self.tree.predict(X)

    def get_params(self, deep=True):
        return {
            "feature_types": self.tree._feature_types,
            "max_depth": self.tree._max_depth,
            "min_samples_split": self.tree._min_samples_split
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter == "feature_types":
                self.tree._feature_types = value
            elif parameter == "max_depth":
                self.tree._max_depth = value
            elif parameter == "min_samples_split":
                self.tree._min_samples_split = value
        return self