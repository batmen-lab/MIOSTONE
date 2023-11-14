from collections import Counter

import numpy as np
import pandas as pd
import torch
from ete3 import Tree, TreeStyle
from scipy.stats.mstats import gmean
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from torch.utils.data import Dataset


class CustomTree(Tree):
    def __init__(self, name, dist, depth, branch_length):
        super().__init__(name=name, dist=dist)
        self.depth = depth
        self.branch_length = branch_length

    def _delete_and_prevent_monochotomy(self):
        parent = self.up
        if not parent:
            return

        for ch in self.children:
            ch.depth = self.depth
            parent.add_child(ch)
           
        parent.remove_child(self)

        if len(parent.children) == 0:
            parent._delete_and_prevent_monochotomy()

    @classmethod
    def from_newick(cls, filename):
        # Read Newick file
        ete_tree = Tree(filename, format=1)
        # Convert ETE Tree to TreeSampler
        tree_sampler = cls._convert_ete_tree_to_custom_tree(ete_tree, 0, 0)
        tree_sampler.max_depth = int(tree_sampler.get_farthest_leaf(topology_only=True)[1]) + 1
        tree_sampler.is_tree_with_equal_branches = True if all(leaf.depth == tree_sampler.max_depth for leaf in tree_sampler.get_leaves()) else False
        return tree_sampler

    @staticmethod
    def _convert_ete_tree_to_custom_tree(ete_tree, depth, branch_length):
        tree_sampler = CustomTree(ete_tree.name, dist=ete_tree.dist, depth=depth, branch_length=branch_length)
        for ete_child in ete_tree.children:
            tree_sampler_child = CustomTree._convert_ete_tree_to_custom_tree(ete_child, depth + 1, branch_length + ete_child.dist)
            tree_sampler.add_child(tree_sampler_child, name=tree_sampler_child.name, dist=ete_child.dist)
        return tree_sampler
    
    def prune_by_dataset(self, dataset, prevent_monochotomy):
        # Extract feature names from dataset
        dataset_features = dataset.X.columns.tolist()

        # Convert to set for fast membership checking
        dataset_feature_set = set(dataset_features)
        
        # Remove leaves that are not in the dataset
        for leaf in self.get_leaves():
            if leaf.name not in dataset_feature_set:
                if prevent_monochotomy:
                    leaf._delete_and_prevent_monochotomy()
                else:
                    leaf.delete()

    def _get_closest_existing_leaves(self, leaf, dataset_feature_set, memo):
        prev_node = leaf 
        current_node = leaf.up

        # Traverse up the tree until a leaf is found in the dataset
        while current_node:
            for child in current_node.children:
                if child == prev_node: 
                    continue

                # Check memoization
                if child in memo:
                    closest_existing_leaves = memo[child]
                    if closest_existing_leaves:  # If a leaf was found for this child
                        return closest_existing_leaves, memo
                    else:  # If no leaf was found for this child in previous searches
                        continue

                memo[child] = []  # Initialize memoization for this child
                for leaf_name in child.iter_leaves():
                    if leaf_name.name in dataset_feature_set:
                        memo[child].append(leaf_name.name) # Memoize the leaf name

                if memo[child]:  # If a leaf was found for this child
                    return memo[child], memo

            prev_node = current_node
            current_node = current_node.up

        # If no closest leaf found, return empty list
        return [], memo

    def augment_dataset_by_tree(self, dataset):
         # Extract feature names from the dataset
        dataset_features = dataset.X.columns.tolist()

        # Convert to set for fast membership checking
        dataset_feature_set = set(dataset_features)

        # Find the missing leaves from the tree
        missing_leaves = [leaf for leaf in self.get_leaves() if leaf.name not in dataset_feature_set]

        memo = {}  # For memoization of leaf names by their parent nodes

        missing_data_dict = {}  # Dictionary to store data for missing leaves

         # Process the missing leaves
        for missing_leaf in missing_leaves:
            # Find closest leaves in the dataset
            closest_existing_leaves, memo = self._get_closest_existing_leaves(missing_leaf, dataset_feature_set, memo)
            
            # If closest leaves found, take the mean of their data. Otherwise, set to 0
            if closest_existing_leaves:
                missing_data_dict[missing_leaf.name] = dataset.X[closest_existing_leaves].mean(axis=1)
            else:
                missing_data_dict[missing_leaf.name] = [0] * len(dataset)

        # Convert the dictionary to a DataFrame and concatenate
        missing_data_df = pd.DataFrame(missing_data_dict, index=dataset.X.index)
        dataset.X = pd.concat([dataset.X, missing_data_df], axis=1)

        return dataset

    def prune_by_dist(self, num_leaves, prevent_monochotomy, topology_only=True, strategy='longest'):
        key = lambda node: node.depth if topology_only else node.branch_length
        if strategy == 'longest':
            # Sort leaves by depth in ascending order
            sorted_leaves = sorted(self.get_leaves(), key=key)
        elif strategy == 'shortest':
            # Sort leaves by depth in ascending order
            sorted_leaves = sorted(self.get_leaves(), key=key, reverse=True)
        else:
            raise ValueError("Unknown order. Available options are 'longest' or 'shortest'")

        # Keep the last num_leaves leaves
        keep_leaves = [leaf.name for leaf in sorted_leaves[:num_leaves]]

        # Prune all other leaves
        for leaf in self.get_leaves():
            if leaf.name not in keep_leaves:
                if prevent_monochotomy:
                    leaf._delete_and_prevent_monochotomy()
                else:
                    leaf.delete()

    def collapse_by_dist_and_merge_features(self, dataset, threshold, topology_only=False):
        to_collapse = {}
        for node in self.traverse("levelorder"):
            children_copy = list(node.children)
            for child in children_copy:
                child_dist = child.depth if topology_only else child.branch_length
                if child_dist > threshold:
                    # get all leaves in the subtree rooted at child
                    leaves = child.get_leaf_names()
                    # detach the child node and its entire subtree
                    child.detach()
                    # store the nodes that need to be collapsed
                    to_collapse[node.name] = leaves

        new_columns = {}
        for parent_name, leaves in to_collapse.items():
            for leaf in leaves:
                if leaf in dataset.X.columns:
                    if parent_name not in new_columns:
                        new_columns[parent_name] = np.exp(dataset.X[leaf])
                    else:
                        new_columns[parent_name] += np.exp(dataset.X[leaf])
            new_columns[parent_name] = np.log(new_columns[parent_name])
            # remove the columns that have been merged
            dataset.X = dataset.X.drop(columns=[x for x in leaves if x in dataset.X.columns])

        # Concatenate the original DataFrame with new columns
        dataset.X = pd.concat([dataset.X, pd.DataFrame(new_columns)], axis=1)
        # Reorder columns 
        dataset.X = dataset.X[self.get_leaf_names()]

        return dataset

    def plot(self):
        # copy tree
        tree = self.copy()

        # set TreeStyle
        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.show_scale = False
        ts.mode = "c"

        # Show the tree
        tree.show(tree_style=ts)

class CustomDataset(Dataset):
    def __init__(self, X, y, classification, prob_label, encode_labels, non_prob_y=None):
        self.classification = classification
        self.prob_label = prob_label
        self.X, self.y = X, y
        self.non_prob_y = non_prob_y

        # Encode labels
        if self.classification and encode_labels:
            self.le = LabelEncoder()
            y_np = self.le.fit_transform(y)
            self.non_prob_y = pd.Series(y_np, name=y.name)
            if self.prob_label:
                y_np = np.eye(len(np.unique(y_np)))[y_np]
                self.y = pd.DataFrame(y_np, columns=self.le.classes_)
            else:
                self.y = self.non_prob_y
            

        # Set number of classes
        if self.classification:
            self.num_classes = len(self.y.columns) if self.prob_label else len(np.unique(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = torch.tensor(self.X.values[idx]).float()
        y = torch.tensor(self.y.values[idx]).long() if self.classification and not self.prob_label else torch.tensor(self.y.values[idx]).float()
        return X, y
    
    def preprocess(self, preprocess_type):
        # Convert to numpy for faster computation
        X_np = self.X.values
        
        # Compute the log geometric mean
        log_gmean = np.log(gmean(X_np.T + 1))

        if preprocess_type == 'clr': # Centre log ratio transformation
            X_np = (np.log(X_np.T + 1) - log_gmean).T
        else: # Log transformation
            X_np = (np.log(X_np.T + 1)).T

        # Convert back to pandas DataFrame
        X = pd.DataFrame(X_np, columns=self.X.columns)    

        return CustomDataset(X, self.y, self.classification, self.prob_label, encode_labels=False)
        
    
    def upsample(self, train_ids, val_ids, factor=1):
        # Convert to numpy for faster computation
        X_np = self.X.values
        y_np = self.y.values

        # Calculate the class distribution only for the samples in ids
        label_data = np.argmax(y_np[train_ids], axis=1) if self.prob_label else y_np[train_ids]
        label_counts = Counter(label_data)

        # Identify the maximum count to up-sample other classes to this count
        max_count = max(label_counts.values()) * factor

        upsampled_X_list = []
        upsampled_y_list = []

        for label, count in label_counts.items():
            # Get the indices for this label
            ids_for_this_label = np.where(label_data == label)[0]
                
            # Map back to the original indices
            ids_for_this_label = train_ids[ids_for_this_label]

            # Only up-sample if there is a shortfall
            shortfall = int(max_count - count)
            if shortfall > 0:
                # Resample data for this label
                X_resampled = resample(X_np[ids_for_this_label], n_samples=max_count)
                y_resampled = resample(y_np[ids_for_this_label], n_samples=max_count)

                upsampled_X_list.append(X_resampled)
                upsampled_y_list.append(y_resampled)
            else:
                upsampled_X_list.append(X_np[ids_for_this_label])
                upsampled_y_list.append(y_np[ids_for_this_label])

        # Combine the original and resampled data
        upsampled_X = np.vstack(upsampled_X_list + [X_np[val_ids]])
        if self.prob_label:
            upsampled_y = np.vstack(upsampled_y_list + [y_np[val_ids]])
        else:
            upsampled_y = np.hstack(upsampled_y_list + [y_np[val_ids]])

        # Convert back to pandas DataFrame
        upsampled_X = pd.DataFrame(upsampled_X, columns=self.X.columns)
        if self.prob_label:
            upsampled_y = pd.DataFrame(upsampled_y, columns=self.y.columns)
        else:
            upsampled_y = pd.Series(upsampled_y, name=self.y.name)

        # Create new ids
        new_train_ids = np.arange(sum([len(x) for x in upsampled_X_list]))
        new_val_ids = np.arange(new_train_ids[-1] + 1, new_train_ids[-1] + 1 + len(val_ids))

        return CustomDataset(upsampled_X, upsampled_y, self.classification, self.prob_label, encode_labels=False), new_train_ids, new_val_ids
    
    def mixup(self, train_ids, val_ids, num_samples, alpha, use_remix, tau, kappa):
        # Convert to numpy for faster computation
        X_np = self.X.values
        y_np = self.y.values

        # Convert to one-hot encoding if the labels are categorical
        if self.classification and not self.prob_label:
            y_oh = np.eye(len(np.unique(y_np)))[y_np]
        else:
            y_oh = y_np

        # Create mixed samples
        mixed_X_list = []
        mixed_y_list = []

        for _ in range(num_samples):
            idx1, idx2 = np.random.choice(train_ids, size=2, replace=False)
            lam = np.random.beta(alpha, alpha)
            mixed_x = lam * X_np[idx1] + (1 - lam) * X_np[idx2]

            if self.classification:
                # Use Remix to generate labels if required
                if use_remix:
                    ni = np.sum(y_np == y_np[idx1])
                    nj = np.sum(y_np == y_np[idx2])

                    if ni/nj >= kappa and lam < tau:
                        lam_y = 0
                    elif ni/nj <= 1/kappa and 1 - lam < tau:
                        lam_y = 1
                    else:
                        lam_y = lam
                else:
                    lam_y = lam
                mixed_y = lam_y * y_oh[idx1] + (1 - lam_y) * y_oh[idx2]
                if not self.prob_label:
                    mixed_y = np.argmax(mixed_y)
            else:
                mixed_y = lam_y * y_np[idx1] + (1 - lam_y) * y_np[idx2]

            mixed_X_list.append(mixed_x)
            mixed_y_list.append(mixed_y)

        X_np = np.vstack([X_np[train_ids]] + mixed_X_list + [X_np[val_ids]])
        if self.prob_label:
            y_np = np.vstack([y_np[train_ids]] + mixed_y_list + [y_np[val_ids]])
        else:
            y_np = np.concatenate([y_np[train_ids], mixed_y_list, y_np[val_ids]])

        # Convert back to pandas DataFrame
        X = pd.DataFrame(X_np, columns=self.X.columns)
        if self.prob_label:
            y = pd.DataFrame(y_np, columns=self.y.columns)
        else:
            y = pd.Series(y_np, name=self.y.name)

        # Create new ids for mixed samples
        new_train_ids = np.arange(len(train_ids) + num_samples)
        new_val_ids = np.arange(new_train_ids[-1] + 1, new_train_ids[-1] + 1 + len(val_ids))

        return CustomDataset(X, y, self.classification, self.prob_label, encode_labels=False), new_train_ids, new_val_ids