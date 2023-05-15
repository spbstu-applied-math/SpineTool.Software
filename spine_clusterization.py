from spine_fitter import SpineFitter
from spine_metrics import SpineMetricDataset
from typing import List, Tuple, Union, Callable, Set
from ipywidgets import widgets
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
import json
from scipy.special import kl_div


class SpineClusterizer(SpineFitter, ABC):
    metric: Union[Callable[[np.ndarray, np.ndarray], float], str]
    _data: np.ndarray
    _labels: List[int]

    def __init__(self, metric: Union[Callable, str] = "euclidean", dim: int = -1, reduction: str = ""):
        super().__init__(dim, reduction)
        self._labels = []
        self.metric = metric

    @property
    def clusters(self) -> List[Set[str]]:
        return list(self.grouping.groups.values())

    @property
    def outlier_cluster(self) -> Set[str]:
        return self.grouping.outlier_group

    @property
    def num_of_clusters(self) -> int:
        return self.grouping.num_of_groups


class SKLearnSpineClusterizer(SpineClusterizer, ABC):
    _fit_data: object
    _clusterizer: object

    def _fit(self, data: np.array, names: List[str]) -> None:
        self._fit_data = self._sklearn_fit(data)
        self._labels = self._fit_data.labels_

        for cluster_index in set(self._labels):
            if cluster_index == -1:
                continue
            names_array = np.array(names)
            cluster_names = names_array[self._labels == cluster_index]
            self.grouping.groups[str(cluster_index + 1)] = set(cluster_names)

    @abstractmethod
    def _sklearn_fit(self, data: np.ndarray) -> object:
        pass


def ks_test(x: np.ndarray, y: np.ndarray) -> float:
    output = 0
    sum_x = 0
    sum_y = 0
    for i in range(x.size):
        sum_x += x[i]
        sum_y += y[i]
        output = max(output, abs(sum_x - sum_y))

    return output


def chi_square_distance(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * np.sum(((x - y) ** 2) / (x + y))


def symmetric_kl_div(x: np.ndarray, y: np.ndarray) -> float:
    x += np.ones_like(x) * 0.001
    x /= np.sum(x)
    y += np.ones_like(x) * 0.001
    y /= np.sum(x)

    a = kl_div(x, y)
    b = kl_div(y, x)

    s = np.sum((a + b) / 2)
    return float(s) / x.size


class DBSCANSpineClusterizer(SKLearnSpineClusterizer):
    eps: float
    min_samples: int

    def __init__(self, eps: float = 0.5, min_samples: int = 2,
                 metric: Union[str, Callable] = "euclidean", dim: int = -1, reduction: str = ""):
        super().__init__(metric=metric, dim=dim, reduction=reduction)
        self.metric = metric
        self.min_samples = min_samples
        self.eps = eps

    def _sklearn_fit(self, data: np.array) -> object:
        self._clusterizer = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        clusterized = self._clusterizer.fit(data)
        return clusterized

class KMeansSpineClusterizer(SKLearnSpineClusterizer):
    _num_of_clusters: int

    def __init__(self, num_of_clusters: int, dim: int = -1, metric="euclidean", reduction: str = ""):
        super().__init__(dim=dim, metric=metric, reduction=reduction)
        self._num_of_clusters = num_of_clusters

    def _sklearn_fit(self, data: np.array) -> object:
        self._clusterizer = KMeans(n_clusters=self._num_of_clusters, random_state=0)
        return self._clusterizer.fit(data)
