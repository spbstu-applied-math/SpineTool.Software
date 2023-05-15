from sklearn.manifold import TSNE

from spine_metrics import SpineMetricDataset
from typing import List, Tuple, Set, Dict, Iterable, Callable, Union
from ipywidgets import widgets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
import json
import random
from scipy.spatial.distance import euclidean
import csv


class IntersectionRatios(Dict[str, Dict[str, float]]):
    @property
    def a_group_labels(self) -> Set[str]:
        return set(self.keys())

    @property
    def b_group_labels(self) -> Set[str]:
        a_group_labels = self.a_group_labels
        if len(a_group_labels) == 0:
            return set()
        return set(self[a_group_labels.pop()].keys())

    @property
    def ordered_a_group_labels(self) -> List[str]:
        output = list(self.a_group_labels)
        output.sort()
        return output

    @property
    def ordered_b_group_labels(self) -> List[str]:
        output = list(self.b_group_labels)
        output.sort()
        return output

    def save(self, filename: str) -> None:
        with open(filename, "w") as file:
            if len(self) == 0:
                return
            denominator_group_key = "denominator_group"
            # do this weird thing with labels because numbers don't sort well as strings
            # it would be 1 11 2 3 ... otherwise
            writer = csv.DictWriter(file, [denominator_group_key] + list(self[self.a_group_labels.pop()].keys()))
            # VV uncomment this line if you figure out a better way to sort VV
            # writer = csv.DictWriter(file, [denominator_group_key] + self.ordered_b_group_labels)
            writer.writeheader()
            for a_group_label, group_ratios in self.items():
                group_ratios: Dict[str, Union[str, float]] = group_ratios.copy()
                group_ratios[denominator_group_key] = a_group_label
                writer.writerow(group_ratios)


class SpineGrouping:
    groups: Dict[str, Set[str]]
    samples: Set[str]
    outliers_label: str

    def __init__(self, samples: Iterable[str] = None, groups: Dict[str, Set[str]] = None,
                 outliers_label: str = None, show_method: str = "tsne"):
        if groups is None:
            groups = {}
        if samples is None:
            samples = set()
            for group in groups.values():
                samples = samples.union(group)
        self.samples = set(samples)
        self.groups = groups
        self.outliers_label = outliers_label
        self._show_method = show_method

    def set_show_method(self, method: str):
        if method == "pca" or method == "tsne":
            self._show_method = method

    @property
    def num_of_groups(self) -> int:
        return len(self.groups)

    @property
    def group_labels(self) -> Set[str]:
        return set(self.groups.keys())

    @property
    def sorted_group_labels(self) -> List[str]:
        labels = list(self.groups.keys())
        labels.sort()
        return labels

    @property
    def outlier_group(self) -> Set[str]:
        ng = self.samples
        for group in self.groups.values():
            ng = ng.difference(group)
        return ng

    @property
    def num_of_outlier(self) -> int:
        return len(self.samples) - len(self.outlier_group)

    @property
    def sample_size(self) -> int:
        return len(self.samples)

    @property
    def colors(self) -> Dict[str, Tuple[float, float, float, float]]:
        label_each = zip(self.groups.keys(),
                         np.linspace(0, 1, self.num_of_groups))
        return {group_label: plt.cm.Spectral(each)
                for (group_label, each) in label_each}

    @property
    def groups_with_outliers(self) -> Dict:
        output = self.groups.copy()
        output[self.outliers_label] = self.outlier_group
        return output

    @property
    def group_labels_with_outliers(self) -> Set[str]:
        output = self.group_labels
        output.add(self.outliers_label)
        return output

    @property
    def colors_with_outliers(self) -> Dict[str, Tuple[float, float, float, float]]:
        output = self.colors
        output[self.outliers_label] = (0.3, 0.3, 0.3, 1)
        return output

    def get_group_size(self, group_label: str) -> int:
        if group_label == self.outliers_label:
            return len(self.outlier_group)
        return len(self.groups[group_label])

    def get_sorted_group(self, group_label: str) -> List[str]:
        spine_names = list(self.groups[group_label])
        spine_names.sort()
        return spine_names

    def get_spines_subset(self, spine_names: Iterable[str]) -> "SpineGrouping":
        spine_names = set(spine_names).intersection(self.samples)
        groups = {label: set() for label in self.group_labels}
        for spine in spine_names:
            label = self.get_group(spine)
            if label != self.outliers_label:
                groups[label].add(spine)
        return SpineGrouping(spine_names, groups, self.outliers_label)

    def get_groups_subset(self, group_labels: Iterable[str]) -> "SpineGrouping":
        groups = {}
        for label in group_labels:
            groups[label] = self.groups[label]
        return SpineGrouping(groups=groups)

    def remove_samples(self, samples_to_removes: Set[str]):
        for sample in samples_to_removes:
            label = self.get_group(sample)
            if label != self.outliers_label:
                self.groups[label].remove(sample)
        self.samples = self.samples.difference(samples_to_removes)

    @staticmethod
    def get_contested_samples(groupings: Iterable["SpineGrouping"], can_vote_outlier: bool = False) -> Set[str]:
        merged = SpineGrouping()

        # merged samples is union of samples from each grouping
        merged.samples = set().union(*[grouping.samples for grouping in groupings])

        # merged grouping contains all groups from each grouping
        for grouping in groupings:
            for label in grouping.groups.keys():
                merged.groups[label] = set()

        contested = set()

        # determine group for each sample
        if len(merged.groups) > 0:
            for spine_name in merged.samples:
                votes = {label: 0 for label in merged.group_labels_with_outliers}
                for grouping in groupings:
                    label = grouping.get_group(spine_name)
                    if label != grouping.outliers_label or can_vote_outlier:
                        votes[label] += 1
                votes_sorted = [(label, vote_num) for (label, vote_num) in votes.items()]
                votes_sorted.sort(key=lambda label_vn: label_vn[1], reverse=True)
                if votes_sorted[0][1] == votes_sorted[1][1]:
                    contested.add(spine_name)

        return contested

    @staticmethod
    def accuracy(true_grouping: "SpineGrouping", predicted_grouping: "SpineGrouping") -> float:
        correct = 0
        for spine in true_grouping.samples:
            if true_grouping.get_group(spine) == predicted_grouping.get_group(spine):
                correct += 1
        return correct / true_grouping.sample_size

    @staticmethod
    def per_group_accuracy(true_grouping: "SpineGrouping", predicted_grouping: "SpineGrouping") -> Dict[str, float]:
        return {label: SpineGrouping.accuracy(true_grouping.get_groups_subset({label}),
                                              predicted_grouping.get_groups_subset({label}))
                for label in true_grouping.group_labels}

    @staticmethod
    def merge(groupings: Iterable["SpineGrouping"], can_vote_outlier: bool = False,
              outliers_label: str = None) -> "SpineGrouping":
        merged = SpineGrouping(outliers_label=outliers_label)

        # merged samples is union of samples from each grouping
        merged.samples = set().union(*[grouping.samples for grouping in groupings])

        # merged grouping contains all groups from each grouping
        for grouping in groupings:
            for label in grouping.groups.keys():
                merged.groups[label] = set()

        # determine group for each sample
        if len(merged.groups) > 0:
            for spine_name in merged.samples:
                votes = {label: 0 for label in merged.group_labels_with_outliers}
                for grouping in groupings:
                    label = grouping.get_group(spine_name)
                    if label != grouping.outliers_label or can_vote_outlier:
                        votes[label] += 1
                votes_sorted = [(label, vote_num) for (label, vote_num) in votes.items()]
                votes_sorted.sort(key=lambda label_vn: label_vn[1], reverse=True)
                most_voted_label = votes_sorted[0][0]
                if most_voted_label != merged.outliers_label:
                    merged.groups[most_voted_label].add(spine_name)

        return merged

    def save(self, filename: str) -> None:
        with open(filename, "w") as file:
            json.dump({"groups": {label: list(group) for (label, group) in self.groups.items()},
                       "samples": list(self.samples), "outliers_label": self.outliers_label}, file)

    def load(self, filename: str) -> "SpineGrouping":
        with open(filename) as file:
            loaded = json.load(file)
            self.samples = set(loaded["samples"])
            self.groups = loaded["groups"]
            if "outliers_label" in loaded:
                self.outliers_label = loaded["outliers_label"]
            else:
                self.outliers_label = None
            for (key, group) in self.groups.items():
                self.groups[key] = set(group)
        return self

    def get_group(self, spine_name: str) -> str:
        for group_label, group in self.groups.items():
            if spine_name in group:
                return group_label
        return self.outliers_label

    def get_color(self, spine_name: str) -> Tuple[float, float, float, float]:
        group_label = self.get_group(spine_name)
        if group_label != self.outliers_label:
            return self.colors[group_label]
        return 0.0, 1, 1, 1

    def show(self, metrics: SpineMetricDataset,
             groups_to_show: Set[int] = None) -> widgets.Widget:
        out = widgets.Output()
        with out:
            self._show(metrics, groups_to_show)
            plt.show()

        return out

    def save_plot(self, metrics: SpineMetricDataset, filename: str) -> None:
        self._show(metrics)
        plt.savefig(filename)
        plt.clf()

    def _show(self, metrics: SpineMetricDataset, groups_to_show: Set[str] = None) -> None:
        def show_group(group_label: str, group: Set[str],
                       color: Tuple[float, float, float, float]) -> None:
            xy = reduced_data[[name_to_index[name] for name in group]]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(color),
                    markeredgecolor="k",
                    markersize=14,
                    label=f"{group_label}"
                )

        if groups_to_show is None:
            groups_to_show = set(self.groups.keys())

        colors = self.colors

        if metrics.as_array().shape[1] > 2:
            metrics = metrics.reduce(2, self._show_method)

        reduced_data = metrics.as_array()
        name_to_index = {name: i for i, name in enumerate(metrics.ordered_spine_names)}

        for (group_label, group) in self.groups.items():
            color = colors[group_label] if group_label in groups_to_show else [
                0.69, 0.69, 0.69, 1]
            show_group(group_label, group, color)
        show_group(self.outliers_label, self.outlier_group, (0, 0, 0, 1))

        plt.title(f"Number of groups: {self.num_of_groups}")
        plt.legend()
        plt.xlabel(metrics.metric_names[0])
        plt.ylabel(metrics.metric_names[1])

    def get_balanced_subset(self, size_ratio: Union[float, Dict] = 0.5) -> "SpineGrouping":
        new_groups = {}
        if isinstance(size_ratio, float):
            size_ratio = {label: size_ratio for label in self.groups.keys()}
        for (label, group) in self.groups.items():
            if len(group) > 0:
                list_group = list(group)
                list_group.sort()
                random.shuffle(list_group)
                new_groups[label] = set(list_group[:int(len(group) * size_ratio[label]) + 1])
            else:
                new_groups[label] = set()

        new_samples = set()
        for group in new_groups.values():
            new_samples = new_samples.union(group)
        outliers = self.outlier_group
        new_samples = new_samples.union(list(outliers)[:int(len(outliers) * np.mean(list(size_ratio.values()))) + 1])

        return SpineGrouping(new_samples, new_groups)

    def intersection_ratios(self, other: "SpineGrouping", normalize: bool = True) -> IntersectionRatios:
        intersections = IntersectionRatios()
        for i, (self_label, self_group) in enumerate(self.groups_with_outliers.items()):
            if len(self_group) == 0:
                continue
            intersections[self_label] = {}
            for j, (other_label, other_group) in enumerate(other.groups_with_outliers.items()):
                if len(other_group) == 0:
                    value = 0
                else:
                    value = len(self_group.intersection(other_group)) / len(self_group)
                    if normalize:
                        value /= len(other_group)
                intersections[self_label][other_label] = value
            if normalize:
                intersection_sum = sum(value for value in intersections[self_label].values())
                for other_label in intersections[self_label].keys():
                    intersections[self_label][other_label] /= intersection_sum
        return intersections

    def get_representative_samples(self, metrics: SpineMetricDataset,
                                   num_of_samples: int = 4,
                                   distance: Callable = euclidean) -> Dict[str, List[str]]:
        if distance is None:
            distance = euclidean

        output = {}
        for label, group in self.groups.items():
            num_of_samples = min(num_of_samples, len(group))
            spine_data = [metrics.row_as_array(spine_name) for spine_name in group]
            # calculate group center
            center = np.mean(spine_data, 0)
            # calculate distance to center for each spine in cluster
            distances = {}
            for (data, name) in zip(spine_data, group):
                distances[name] = distance(center, data)
            # sort spines by distance
            sorted_by_distance = list(group)
            sorted_by_distance.sort(key=lambda name: distances[name])
            # return first N spine names
            output[label] = sorted_by_distance[:num_of_samples]

        return output

    def get_metric_distributions(self, metrics: SpineMetricDataset) -> Dict[str, np.array]:
        metric_distributions = {}
        for label, group in self.groups.items():
            metric_distributions[label] = []
            for metric in metrics.row(list(metrics.spine_names)[0]):
                group_metrics = metrics.get_spines_subset(group)
                metric_column = group_metrics.column(metric.name).values()
                metric_distributions[label].append(metric.get_distribution(metric_column))
        return metric_distributions

    def save_metric_distribution(self, metrics: SpineMetricDataset, filename: str) -> None:
        all_distributions = self.get_metric_distributions(metrics)
        with open(filename, "w") as file:
            writer = csv.writer(file)
            for label, group_distributions in all_distributions.items():
                writer.writerow([label])
                name_distribution = zip(metrics.metric_names, group_distributions)
                for metric_name, metric_distribution in name_distribution:
                    writer.writerow([metric_name] + list(metric_distribution))

    def save_reduced(self, metrics: SpineMetricDataset, filename: str, method: str = "pca") -> None:
        reduced_metrics = metrics.reduce(2, method)


        with open(filename, "w") as file:
            for label, group in self.groups.items():
                # write grouping label
                file.write(f"{label}\n\n")

                # only consider spines from this group
                reduced_metrics_subset = reduced_metrics.get_spines_subset(group)
                
                # write header
                writer = csv.DictWriter(file, [method] + reduced_metrics_subset.ordered_spine_names)
                writer.writeheader()

                # write pca coordinates for every spine
                for reduced_coord_name in reduced_metrics_subset.metric_names:
                    column: Dict = reduced_metrics_subset.column(reduced_coord_name)
                    for key, value in column.items():
                        column[key] = value.value
                    column[method] = reduced_coord_name
                    writer.writerow(column)


class SpineFitter(ABC):
    grouping: SpineGrouping
    dim: int
    reduction: str
    fit_metrics: SpineMetricDataset

    def __init__(self, dim: int = -1, reduction: str = ""):
        assert dim > 0 or not reduction
        self.dim = dim
        self.reduction = reduction
        self.grouping = SpineGrouping()

    def set_show_method(self, method: str = "tsne"):
        self.grouping.set_show_method(method)

    def fit(self, spine_metrics: SpineMetricDataset) -> None:
        self.fit_metrics = spine_metrics
        data = spine_metrics.as_array()
        if self.dim != -1:
            self.fit_metrics = spine_metrics.reduce(self.dim, self.reduction)
            if self.reduction == "pca":
                data = PCA(self.dim).fit_transform(data)
            elif self.reduction == "tsne":
                data = TSNE(self.dim, init="pca").fit_transform(data)
            else:
                raise NotImplemented(f"method {self.reduction} is not supported")

        self.grouping.samples = spine_metrics.spine_names

        self._fit(data, spine_metrics.ordered_spine_names)

    @abstractmethod
    def _fit(self, data: np.array, names: List[str]) -> object:
        pass

    def show(self) -> widgets.Widget:
        return self.grouping.show(self.fit_metrics)
