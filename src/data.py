import importlib
import os

import numpy as np
import pandas as pd
from ete4 import Tree
from sklearn.preprocessing import MinMaxScaler
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
        self.taxonomic_ranks = ["Life", "Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species", "Taxa"]

    @classmethod
    def init_from_nwk(cls, nwk_file):
        """
        Initialize a MIOSTONETree object from a newick file.

        :param nwk_file: File path to the newick file.
        :return: A MIOSTONETree instance.
        """
        t = Tree(open(nwk_file), parser=1)
        t.name = 'root'
        return cls(t)

    def prune(self, features):
        """
        Prune the tree to only include the specified features.

        :param features: A list of features to preserve.
        """
        leaves = set(self.ete_tree.leaves())
        while any(leaf.name not in features for leaf in leaves):
            for leaf in leaves:
                if leaf.name not in features:
                    leaf.delete(prevent_nondicotomic=False)
            leaves = set(self.ete_tree.leaves())

    def compute_depths(self):
        """
        Compute the depths of all nodes in the tree.
        """
        for ete_node in self.ete_tree.traverse("levelorder"):
            if ete_node.is_root:
                self.depths[ete_node.name] = 0
            else:
                self.depths[ete_node.name] = self.depths[ete_node.up.name] + 1
            if self.depths[ete_node.name] == 2 and ete_node.is_leaf:
                print(ete_node.up.up.name)
            self.max_depth = max(self.max_depth, self.depths[ete_node.name])

    def compute_indices(self):
        """
        Compute the indices of all nodes in the tree.
        """
        self.indices = {}
        curr_id = 0
        curr_depth = 0
        for ete_node in self.ete_tree.traverse("levelorder"):
            node_depth = self.depths[ete_node.name]
            if node_depth > curr_depth:
                curr_id = 0
                curr_depth = node_depth
            self.indices[ete_node.name] = curr_id
            curr_id += 1

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

    def __init__(self, X, y, ids, features):
        """
        Initialize the MIOSTONEDataset object.

        :param X: The input features as a numpy array.
        :param y: The target variable as a numpy array.
        :param ids: The sample IDs as a numpy array.
        :param features: The feature names as a numpy array.
        """
        self.X = X
        self.y = y
        self.ids = ids
        self.features = features
        self.num_classes = len(np.unique(y))
        self.class_weight = len(y) / (self.num_classes * np.bincount(y))
        self.normalized = False
        self.clr_transformed = False
        self.standardized = False
        self.one_hot_encoded = False
        self.tree_matrix_repr = False

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
    def init_from_files(cls, data_fp, meta_fp, target_fp):
        """
        Initialize a MIOSTONEDataset object from files.

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
        ids = np.array(ids)
        features = data.index.values

        return cls(X, y, ids, features)
    
    def normalize(self):
        """
        Normalize the dataset by adding 1 to avoid division by zero and then 
        dividing each feature value by the sum of values in its sample.

        :raises ValueError: If the dataset is already normalized.
        """
        if self.normalized:
            raise ValueError("Dataset is already normalized")
        self.X_sum = self.X.sum(axis=1, keepdims=True)
        self.X = self.X / self.X_sum
        self.normalized = True 
    
    def clr_transform(self):
        """
        Apply centered log-ratio (CLR) transformation to the normalized dataset.

        :raises ValueError: If the dataset is already CLR transformed.
        """
        if self.clr_transformed:
            raise ValueError("Dataset is already clr-transformed")
        
        # Apply log transformation
        if self.normalized:
            self.X = np.log(self.X)
        else:
            self.X = np.log1p(self.X)

        # Subtract the mean of each sample
        self.X = self.X - self.X.mean(axis=1, keepdims=True)

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

    def drop_features_by_tree(self, tree):
        """
        Drop features from the dataset based on the tree.
        
        :param tree: A MIOSTONETree instance.
        """
        leaf_names = [leaf for leaf in tree.ete_tree.leaf_names() if leaf in self.features]
        self.X = self.X[:, [self.features.tolist().index(leaf) for leaf in leaf_names]]
        self.features = np.array(leaf_names)

    def add_features_by_tree(self, tree):
        """
        Add new features to the dataset based on the tree. The new features take the
        values of the closest existing feature in the dataset.

        :param tree: A MIOSTONETree instance.
        """
        closest_features = self._find_closest_features(tree)
        if closest_features:
            self._add_new_features(closest_features)
        
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
                closest_feature = self._find_closest_by_common_ancestor(leaf, existing_features)
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
            for desc in current_node.descendants():
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
            self.features = np.append(self.features, new_feature)

        new_features_data = np.array(new_features_data).T
        self.X = np.column_stack((self.X, new_features_data))
        self.features = np.append(self.features, list(closest_features.keys()))

    def order_features_by_tree(self, tree):
        """
        Order the features in the dataset by the tree. 

        :param tree: A MIOSTONETree instance.
        """
        leaf_names = list(tree.ete_tree.leaf_names())
        self.X = self.X[:, [self.features.tolist().index(leaf) for leaf in leaf_names]]
        self.features = np.array(leaf_names)

    def to_popphycnn_matrix(self, tree, scaler=None):
        """
        Convert the dataset to a PopPhyCNN matrix representation.

        :param tree: A MIOSTONETree instance.
        :param scaler: A Scaler instance to apply to the matrix. Defaults to None.
        :return: A Scaler instance used to scale the matrix.
        :raises ValueError: If the dataset is already in PopPhyCNN matrix representation or if the dataset is CLR transformed.
        """
        if self.tree_matrix_repr:
            raise ValueError("Dataset is already in PopPhyCNN matrix representation")
        if self.clr_transformed:
            raise ValueError("Dataset must not be clr-transformed before converting to PopPhyCNN matrix representation")

        # Create a matrix with the same shape as the tree
        num_rows = tree.max_depth + 1
        num_cols = len(list(tree.ete_tree.leaves()))
        M = np.zeros((self.X.shape[0], num_rows, num_cols), dtype=np.float32)

        # Iterate over the nodes in the tree in level order and fill in the matrix
        for ete_node in reversed(list(tree.ete_tree.traverse("levelorder"))):
            if ete_node.is_leaf:
                M[:, tree.depths[ete_node.name], tree.indices[ete_node.name]] = self.X[:, tree.indices[ete_node.name]]
            else:
                for child in ete_node.get_children():
                    M[:, tree.depths[ete_node.name], tree.indices[ete_node.name]] += M[:, tree.depths[child.name], tree.indices[child.name]]

        # Apply log transformation
        M = np.log1p(M)

        # Apply scaling to the matrix
        if scaler:
            M = scaler.transform(M.reshape(-1, num_rows * num_cols)).reshape(-1, num_rows, num_cols)
        else:
            scaler = MinMaxScaler()
            M = scaler.fit_transform(M.reshape(-1, num_rows * num_cols)).reshape(-1, num_rows, num_cols)
        M = np.clip(M, 0, 1)

        self.X = M
        self.tree_matrix_repr = True

        return scaler