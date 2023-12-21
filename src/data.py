import copy
import importlib
import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MIOSTONETree:
    """
    A class to represent the hierarchical structure of features in MIOSTONE datasets.

    Attributes:
        ete_tree (ete4.Tree): An ete4 Tree instance.
        depths (dict): A dictionary mapping feature names to their depths in the tree.
        max_depth (int): The maximum depth of the tree.
    """

    def __init__(self, ete_tree):
        """
        Initialize the MIOSTONETree object.

        :param tree: An ete4 Tree instance.
        """
        self.ete_tree = ete_tree
        self.depths = {}
        self.max_depth = 0
        self._compute_depths()

    def _compute_depths(self):
        """
        Compute the depths of all features in the tree. This method is used internally
        by the __init__ method.
        """
        for ete_node in self.ete_tree.traverse("levelorder"):
            if ete_node.is_root:
                self.depths[ete_node.name] = 0
            else:
                self.depths[ete_node.name] = self.depths[ete_node.up.name] + 1
            self.max_depth = max(self.max_depth, self.depths[ete_node.name])


class MIOSTONEDataset(Dataset):
    """
    A class to handle MIOSTONE datasets, offering functionality for preprocessing, 
    normalization, CLR transformation, and one-hot encoding of target variables.

    Attributes:
        X (np.array): The input features.
        y (np.array): The target variable.
        features (np.array): The feature names.
        normalized (bool): Whether the dataset is normalized.
        clr_transformed (bool): Whether the dataset is CLR transformed.
        one_hot_encoded (bool): Whether the target variable is one-hot encoded.
    """

    def __init__(self, X, y, features):
        """
        Initialize the MIOSTONEDataset object.

        :param X: The input features as a numpy array.
        :param y: The target variable as a numpy array.
        :param features: The feature names as a numpy array.
        """
        self.X = X
        self.y = y
        self.features = features
        self.num_classes = len(np.unique(y))
        self.class_weight = len(y) / (self.num_classes * np.bincount(y))
        self.normalized = False
        self.clr_transformed = False
        self.one_hot_encoded = False

    def __len__(self):
        """
        Return the number of samples in the dataset.

        :return: The number of samples.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its corresponding target value from the dataset.

        :param idx: The index of the sample to retrieve.
        :return: A tuple containing the sample and its target value.
        """
        return self.X[idx], self.y[idx]
    
    @classmethod
    def preprocess(self, data_fp, meta_fp, target_fp):
        """
        Preprocess the dataset from given file paths.

        :param data_fp: File path to the MIOSTONE data.
        :param meta_fp: File path to the metadata.
        :param target_fp: File path to the target function script.
        :return: Processed feature data (X), target values (y), and feature names.
        :raises ValueError: If the target function file does not exist.
        """
        # read data and metadata
        data = pd.read_table(data_fp, index_col=0)
        meta = pd.read_table(meta_fp, index_col=0)

        # load target function
        if not os.path.exists(target_fp):
            raise ValueError('Target function does not exist.')
        spec = importlib.util.spec_from_file_location('target', target_fp)
        target_lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(target_lib)
        target = target_lib.target

        # get target values and test name
        prop, name = target(meta)

        # filter samples
        ids = [x for x in data.columns if x in set(prop.index)]
        data, prop = data[ids], prop.loc[ids]

        # input data
        X = data.values.T.astype(np.float32)
        y = prop.values
        features = data.index.values

        return X, y, features
    
    def normalize(self):
        """
        Normalize the dataset by adding 1 to avoid division by zero and then 
        dividing each feature value by the sum of values in its sample.

        :raises ValueError: If the dataset is already normalized.
        """
        if self.normalized:
            raise ValueError("Dataset is already normalized")
        self.X = self.X + 1 # Add 1 to avoid division by 0
        self.X = self.X / self.X.sum(axis=1, keepdims=True)
        self.normalized = True 
    
    def clr_transform(self):
        """
        Apply centered log-ratio (CLR) transformation to the normalized dataset.

        :raises ValueError: If the dataset is not normalized or is already CLR transformed.
        """
        if not self.normalized:
            raise ValueError("Dataset must be normalized before clr-transformation")
        if self.clr_transformed:
            raise ValueError("Dataset is already clr-transformed")
        self.X = np.log(self.X) - np.log(self.X.mean(axis=1, keepdims=True))
        self.clr_transformed = True

    def one_hot_encode(self):
        """
        One-hot encode the target variable of the dataset.

        :raises ValueError: If the target variable is already one-hot encoded.
        """
        if self.one_hot_encoded:
            raise ValueError("Dataset is already one-hot encoded")
        self.y = np.eye(len(np.unique(self.y)))[self.y.astype(int)]
        self.one_hot_encoded = True

    def add_features_from_tree(self, tree):
        """
        Add new features to the dataset based on the tree. The new features take the
        values of the closest existing feature in the dataset.

        :param tree: A MIOSTONETree instance.
        """
        closest_features = self._find_closest_features(tree)
        self._add_new_features(closest_features)
        self.order_features_by_tree(tree)

    def _find_closest_features(self, tree):
        """
        Find the closest existing feature in the dataset for each new feature in the tree.
        This method is used internally by the add_features_from_tree method.

        :param tree: A MIOSTONETree instance.
        :return: A dictionary mapping new features to their closest existing features.
        """
        existing_features = set(self.features)
        closest_features = {}

        for leaf in tree.ete_tree.leaves():
            if leaf.name not in existing_features:
                closest_feature = self._find_closest_in_ancestors(leaf, existing_features)
                if closest_feature:
                    closest_features[leaf.name] = closest_feature

        return closest_features

    def _find_closest_by_common_ancestor(self, leaf, existing_features):
        """
        Find the closest existing feature in the dataset for a new feature by finding the
        common ancestor of the new feature and existing features. This method is used
        internally by the add_features_from_tree method.

        :param leaf: The new feature to find the closest existing feature for.
        :param existing_features: The existing features in the dataset.
        :return: The closest existing feature. If no common ancestor is found, return None.
        """
        current_node = leaf
        while current_node.up:
            current_node = current_node.up
            for desc in current_node.get_descendants():
                if desc.name in existing_features:
                    return desc.name
        return None

    def _add_new_features(self, closest_features):
        """
        Add new features to the dataset based on the closest features identified. 
        This method is used internally by the add_features_from_tree method.

        :param closest_features: A dictionary mapping new features to their closest existing features.
        """
        new_features_data = []
        for new_feature, closest_feature in closest_features.items():
            closest_feature_index = np.where(self.features == closest_feature)[0][0]
            new_feature_values = self.X[:, closest_feature_index]
            new_features_data.append(new_feature_values)

        self.X = np.column_stack((self.X, new_features_data))

    def order_features_by_tree(self, tree):
        """
        Order the features in the dataset by the tree. 

        :param tree: A MIOSTONETree instance.
        """
        leaf_names = list(tree.ete_tree.leaf_names())
        self.X = self.X[:, [self.features.tolist().index(leaf) for leaf in leaf_names]]
        self.features = np.array(leaf_names) 
        
class MIOSTONEMixup:
    """
    A class for applying Mixup and Cutmix augmentations to MIOSTONE datasets using a tree-based structure. 
    It handles the mixing of samples based on the Aitchison distance and tree-based relationships.

    Attributes:
        dataset (MIOSTONEDataset): The MIOSTONE dataset to be augmented.
        tree (MIOSTONETree): A tree representing the hierarchical structure of features in the dataset.
        alpha (float): The alpha parameter for the Beta distribution used in Mixup.
    """
    def __init__(self, dataset, tree, alpha=0.4):
        """
        Initialize the MIOSTONEMixup object.

        :param dataset: MIOSTONEDataset instance for the dataset to augment.
        :param tree: MIOSTONETree instance representing the feature hierarchy.
        """
        self.dataset = copy.deepcopy(dataset)
        self.tree = tree

        # Ensure that the dataset is normalized and one-hot encoded, but not clr-transformed
        if not self.dataset.normalized:
            self.dataset.normalize()
        if not self.dataset.one_hot_encoded:
            self.dataset.one_hot_encode()
        if self.dataset.clr_transformed:
            raise ValueError("Dataset must not be clr-transformed")

        # Compute distances
        self.distances = self._compute_aitchison_distances()

    def _compute_aitchison_distances(self):
        """
        Compute the Aitchison distances between all pairs of samples in the dataset.
        This method is used internally by the __init__ method.
        """
        # Aitchison distance is the Euclidean distance between the clr-transformed samples
        distances = np.zeros((len(self.dataset), len(self.dataset)))
        for i in range(len(self.dataset)):
            for j in range(i+1, len(self.dataset)):
                distances[i, j] = self._aitchison_distance(self.dataset.X[i], self.dataset.X[j])
        return distances
    
    def _compute_eligible_pairs(self, min_threshold, max_threshold):
        """
        Compute the eligible sample pairs for Mixup based on the Aitchison distance thresholds.
        This method is used internally by the _select_samples method.
        
        :param min_threshold: A float representing the minimum Aitchison distance threshold.
        :param max_threshold: A float representing the maximum Aitchison distance threshold.
        """
        eligible_pairs = []
        for i in range(len(self.dataset)):
            for j in range(i+1, len(self.dataset)):
                if self.distances[i, j] >= min_threshold and self.distances[i, j] <= max_threshold:
                    eligible_pairs.append((i, j))
        return eligible_pairs
    
    def _select_samples(self, eligible_pairs, num_samples):
        """
        Select sample pairs for Mixup without replacement. 
        This method is used internally by the mixup method.

        :param eligible_pairs: A list of eligible sample pairs for Mixup.
        :param num_samples: The number of Mixup samples to generate.
        :return: A list of selected sample pairs.
        :raises ValueError: If no sample pairs are found within the distance threshold.
        """
        # Check if there are any eligible pairs
        if not eligible_pairs:
            raise ValueError("No sample pairs found with distance below the threshold")
        
        # Select num_samples pairs without replacement
        selected_pairs = random.sample(eligible_pairs, num_samples)
        return selected_pairs

    def _aitchison_distance(self, x, y):
        """
        Compute the Aitchison distance between two samples. 
        This method is used internally by the _compute_aitchison_distances method.

        :param x: A numpy array representing the first sample.
        :param y: A numpy array representing the second sample.
        :return: The Aitchison distance between the two samples.
        """
        x = np.log(x) - np.log(x).mean()
        y = np.log(y) - np.log(y).mean()
        return np.linalg.norm(x - y, ord=2)
    
    def _aitchison_addition(self, x, v):
        """
        Compute the Aitchison addition between two samples. 
        This method is used internally by the _mixup_samples method.

        :param x: A numpy array representing the first sample.
        :param v: A numpy array representing the second sample.
        :return: The Aitchison addition between the two samples.
        """
        sum_xv = np.sum(x * v)
        return (x * v) / sum_xv

    def _aitchison_scalar_multiplication(self, lam, x):
        """
        Compute the Aitchison scalar multiplication between a scalar and a sample. 
        This method is used internally by the _mixup_samples method.

        :param lam: A float representing the scalar.
        :param x: A numpy array representing the sample.
        :return: The Aitchison scalar multiplication between the scalar and the sample.
        """
        sum_xtolam = np.sum(x ** lam)
        return (x ** lam) / sum_xtolam

    def _mixup_samples(self, idx1, idx2, lam):
        # Select the pair of samples
        x1, y1 = self.dataset[idx1]
        x2, y2 = self.dataset[idx2]

        # Compute Aitchison mixup
        mixed_x = self._aitchison_addition(self._aitchison_scalar_multiplication(lam, x1), self._aitchison_scalar_multiplication(1-lam, x2))
        mixed_y = lam * y1 + (1 - lam) * y2
        return mixed_x, mixed_y

    def mixup(self, min_threshold, max_threshold, num_samples, alpha=0.4):
        """
        Perform Mixup augmentation on the dataset. Mixup is applied to pairs of samples 
        that are within the specified Aitchison distance thresholds.

        :param min_threshold: A float representing the minimum Aitchison distance for selecting sample pairs.
        :param max_threshold: A float representing the maximum Aitchison distance for selecting sample pairs.
        :param num_samples: The number of Mixup samples to generate.
        :param alpha: The alpha parameter for the Beta distribution. Defaults to 0.4.
        :return: An augmented MIOSTONEDataset instance.
        :raises ValueError: If no sample pairs are found within the distance threshold.
        """
        # Compute eligible pairs
        eligible_pairs = self._compute_eligible_pairs(min_threshold, max_threshold)

        # Mixup samples
        mixed_xs, mixed_ys = [], []
        selected_pairs = self._select_samples(eligible_pairs, num_samples)
        for idx1, idx2 in selected_pairs:
            lam = np.random.beta(alpha, alpha)
            mixed_x, mixed_y = self._mixup_samples(idx1, idx2, lam)
            mixed_xs.append(mixed_x)
            mixed_ys.append(mixed_y)

        self.dataset.X = np.vstack((self.dataset.X, np.array(mixed_xs)))
        self.dataset.y = np.vstack((self.dataset.y, np.array(mixed_ys)))

        self.dataset.clr_transform()

        return self.dataset
    
    def _select_subtree(self, num_subtrees):
        """
        Select subtrees from the tree with probability proportional to their depth.
        This method is used internally by the cutmix method.

        :param num_subtrees: The number of subtrees to select.
        :return: A list of selected subtrees.
        """
        # Select a node from the subtree with probability proportional to its depth
        subtree_nodes = []
        available_nodes = list(self.tree.ete_tree.leaves())

        for _ in range(num_subtrees):
            selected_node = random.choices(available_nodes, weights=[self.tree.depths[node.name] for node in available_nodes])[0]
            subtree_nodes.append(selected_node)

            # Remove the selected node and its ancestors and descendants from the available nodes
            for node in selected_node.ancestors():
                if node in available_nodes:
                    available_nodes.remove(node)
            for node in selected_node.descendants():
                if node in available_nodes:
                    available_nodes.remove(node)

        return subtree_nodes
    
    def _cutmix_with_subtree(self, idx1, idx2, subtree_nodes):
        """
        Perform Cutmix on a pair of samples by swapping the data of leaves in the specified subtrees.
        This method is used internally by the cutmix method.

        :param idx1: The index of the first sample.
        :param idx2: The index of the second sample.
        :param subtree_nodes: A list of nodes in the tree representing the subtrees to swap.
        :return: The swapped samples.
        """
        # Select the pair of samples
        x1, y1 = self.dataset[idx1]
        x2, y2 = self.dataset[idx2]

        num_swapped_leaves = 0
        for node in subtree_nodes:
            # Find the index of leaves in the subtree
            leaf_names = [leaf_name for leaf_name in node.leaf_names()]
            leaf_idx = [self.dataset.features.tolist().index(leaf_name) for leaf_name in leaf_names]
            num_swapped_leaves += len(leaf_idx)
            # Swap the data of the tip nodes
            x1_orig, x2_orig = x1[leaf_idx].copy(), x2[leaf_idx].copy()
            x1[leaf_idx], x2[leaf_idx] = x2_orig, x1_orig
            # Rescale the swapped data to ensure that the sum of values is the same as the original
            x1[leaf_idx] = x1[leaf_idx] * x1_orig.sum() / x1[leaf_idx].sum()
            x2[leaf_idx] = x2[leaf_idx] * x2_orig.sum() / x2[leaf_idx].sum()

        # Compute the new labels
        y1 = y1 * (1 - num_swapped_leaves / len(self.dataset.features)) + y2 * (num_swapped_leaves / len(self.dataset.features))
        y2 = y2 * (1 - num_swapped_leaves / len(self.dataset.features)) + y1 * (num_swapped_leaves / len(self.dataset.features))
        
        return x1, y1, x2, y2
    
    def cutmix(self, min_threshold, max_threshold, num_samples, num_subtrees):
        """
        Perform Cutmix augmentation on the dataset. Cutmix is applied by swapping subtrees between pairs 
        of samples that are within the specified Aitchison distance thresholds.
        
        :param min_threshold: A float representing the minimum Aitchison distance for selecting sample pairs.
        :param max_threshold: A float representing the maximum Aitchison distance for selecting sample pairs.
        :param num_samples: The number of Cutmix samples to generate.
        :param num_subtrees: The number of subtrees to swap between each pair of samples.
        :return: An augmented MIOSTONEDataset instance.
        :raises ValueError: If no sample pairs are found within the distance threshold.
        """
        # Compute eligible pairs
        eligible_pairs = self._compute_eligible_pairs(min_threshold, max_threshold)

        # Cutmix samples
        mixed_xs, mixed_ys = [], []
        selected_pairs = self._select_samples(eligible_pairs, num_samples // 2)
        for idx1, idx2 in selected_pairs:
            # Select a subtree
            subtree_nodes = self._select_subtree(num_subtrees)
            # Cutmix the samples
            mixed_x1, mixed_y1, mixed_x2, mixed_y2 = self._cutmix_with_subtree(idx1, idx2, subtree_nodes)
            mixed_xs.append(mixed_x1)
            mixed_ys.append(mixed_y1)
            mixed_xs.append(mixed_x2)
            mixed_ys.append(mixed_y2)

        self.dataset.X = np.vstack((self.dataset.X, np.array(mixed_xs)))
        self.dataset.y = np.vstack((self.dataset.y, np.array(mixed_ys)))

        self.dataset.clr_transform()

        return self.dataset