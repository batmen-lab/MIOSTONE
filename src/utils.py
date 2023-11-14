import glob
import os
import random

import numpy as np
import pandas as pd
import torch
from ete3 import Tree

from data import CustomDataset, CustomTree


def depth2taxo(depth):
    depth2taxo = {
        0: "Life",
        1: "Domain",
        2: "Phylum",
        3: "Class",
        4: "Order",
        5: "Family",
        6: "Genus",
        7: "Species",
        8: "Strain"
    }
    if depth > 8:
        return "Raw Data"
    return depth2taxo[depth]

def load_data(dataset_path, dataset_name):
    data_files = glob.glob(f"{dataset_path}/data*")
    metadata_files = glob.glob(f"{dataset_path}/meta*")

    if len(data_files) == 0 or len(metadata_files) == 0:
        raise FileNotFoundError("Could not find data and metadata files in dataset directory")
    
    data_file = data_files[0]
    metadata_file = metadata_files[0]

    data = pd.read_table(data_file, index_col=0)
    index_col = 1 if dataset_name == "T1D" else 0
    metadata = pd.read_table(metadata_file, index_col=index_col)

    return data, metadata

def load_model(model_path):
    seed, hyperparams = parse_file_name(os.path.basename(model_path))
    model = torch.load(model_path)
    return model, seed, hyperparams

def parse_file_name(file_name):
    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split("_")

    def extract_parameter(prefix, num_params, datatypes):
        indices = [i for i, part in enumerate(parts) if part.startswith(prefix)]
        if indices:
            if len(indices) > 1:
                indices.pop(0)
            for i in range(num_params):
                indices.append(indices[-1] + 1)
            return [datatype(parts[index].replace(prefix, "")) for index, datatype in zip(indices, datatypes)]
        return None
    
    seed = int(parts[0])
    model = parts[1]
    mx_values = extract_parameter("mx", 2, [int, float])
    zs_values = extract_parameter("zs", 1, [int])
    ft_values = extract_parameter("ft", 2, [int, int])
    td_values = extract_parameter("td", 2, [str, str])
    fg_values = extract_parameter("fg", 1, [float])

    if model == "rf":
        hyperparameters = {
            "model": model,
            "preprocess": parts[2],
            "rebalance": parts[3],
            # "collapse_threshold": extract_parameter("c", 1, [float])[0],
            "mixup_num_samples": mx_values[0] if mx_values else 0,
            "mixup_alpha": mx_values[1] if mx_values else 0.0,
            "prob_label": True if "_pl" in base_name else False,
            "taxonomy": True if "_tx" in base_name else False,
            "tree_dataset": td_values[0] if td_values else None,
            "tree_target": td_values[1] if td_values else None,
            "prune_by_dataset": True if "_pd" in base_name else False,
            "transfer_learning": "ft" if zs_values else "none",
            "pretrained_seed": zs_values[0] if zs_values else 0,
            "ft_epochs": 0,
        }
    else:
        if model == "mlp":
            hyperparameters = {
                "model": model,
                "preprocess": parts[2],
                "rebalance": parts[3],
                "batch_size": int(parts[4]),
                "epochs": int(parts[5]),
                "loss": parts[6],
                "mlp_type": parts[7],
                "mixup_num_samples": mx_values[0] if mx_values else 0,
                "mixup_alpha": mx_values[1] if mx_values else 0.0,
                "prob_label": True if "_pl" in base_name else False,
                "taxonomy": True if "_tx" in base_name else False,
                "tree_dataset": td_values[0] if td_values else None,
                "tree_target": td_values[1] if td_values else None,
                "prune_by_dataset": True if "_pd" in base_name else False,
                "focal_gamma": fg_values[0] if fg_values else 0.0,
                "transfer_learning": "ft" if zs_values or ft_values else "none",
                "pretrained_seed": zs_values[0] if zs_values else ft_values[0] if ft_values else 0,
                "ft_epochs": ft_values[1] if ft_values else 0,
            }
        else:
            hyperparameters = {
                "model": model,
                "preprocess": parts[2],
                "rebalance": parts[3],
                "batch_size": int(parts[4]),
                "epochs": int(parts[5]),
                "loss": parts[6],
                "node_min_dim": int(parts[7]),
                "node_dim_func" : parts[8],
                "node_dim_func_param" : float(parts[9]),
                "node_gate_type" : parts[10],
                "node_gate_param" : float(parts[11]),
                "output_depth": int(parts[12]),
                "mixup_num_samples": mx_values[0] if mx_values else 0,
                "mixup_alpha": mx_values[1] if mx_values else 0.0,
                "taxonomy": True if "_tx" in base_name else False,
                "tree_dataset": td_values[0] if td_values else None,
                "tree_target": td_values[1] if td_values else None,
                "prune_by_dataset": True if "_pd" in base_name else False,
                "prob_label": True if "_pl" in base_name else False,
                "output_truncation": True if "_ot" in base_name else False,
                "focal_gamma": fg_values[0] if fg_values else 0.0,
                "transfer_learning": "ft" if zs_values or ft_values else "none",
                "pretrained_seed": zs_values[0] if zs_values else ft_values[0] if ft_values else 0,
                "ft_epochs": ft_values[1] if ft_values else 0,
            }

    return seed, hyperparameters

def preprocess_data(data, metadata, dataset_name, target, phylogeny, taxonomy, prob_label, prune_by_dataset, prune_by_num_leaves=-1, custom_tree=None):
    # Load the phylogenetic/taxonomic tree
    if custom_tree is None:
        if taxonomy:
            newick_file = f"../data/{phylogeny}/taxonomy.nwk"
            if not os.path.exists(newick_file):
                taxonomy_file = glob.glob(f"../data/{phylogeny}/taxonomy*")[0]
                if not os.path.exists(taxonomy_file):
                    raise FileNotFoundError("Could not find taxonomy file in dataset directory")
                taxonomy_to_tree(taxonomy_file, newick_file)
        else:
            newick_file = f"../data/{phylogeny}/phylogeny.nwk"
        custom_tree = CustomTree.from_newick(newick_file)
    else:
        custom_tree = custom_tree

    prevent_monochotomy = True
    
    # Filter the data to only include samples that are in the metadata
    target = 'diagnosis' if target == 'Type' else target
    samples = sorted(set(data.columns).intersection(set(metadata.index)))
    X = data[samples]
    y = metadata.loc[samples][target]

    # Filter the data to exclude samples that have "not applicable" or "not provided" labels
    y = y[~y.isin(["not applicable", "not provided"])]
    X = X[y.index]

    # Rename "cd" to "CD", "uc" to "UC and "healthy_control" to "nonIBD" for IBD200
    if dataset_name == "IBD200":
        y = y.replace("cd", "CD")
        y = y.replace("uc", "UC")
        y = y.replace("healthy_control", "nonIBD")
        # Remove all other classes
        y = y[y.isin(["CD", "UC"])]
        X = X[y.index]

    if dataset_name == "HMP2":
        y = y[y.isin(["CD", "UC"])]
        X = X[y.index]

    # Create the dataset
    dataset = CustomDataset(X.T, y, classification=True, prob_label=prob_label, encode_labels=True)

    # Prune the tree based on the dataset if required
    if prune_by_dataset:
        custom_tree.prune_by_dataset(dataset, prevent_monochotomy)
    else: # Otherwise, augment the dataset by the tree
        dataset = custom_tree.augment_dataset_by_tree(dataset)

    dataset.X = dataset.X[custom_tree.get_leaf_names()]
    assert all(custom_tree.get_leaf_names() == dataset.X.columns)

    # Collapse the tree based on the branch lengths
    '''
    if args.collapse_threshold >= 0:
        dataset = custom_tree.collapse_by_dist_and_merge_features(dataset, args.collapse_threshold)
        assert all(custom_tree.get_leaf_names() == dataset.X.columns)
    '''

    # Prune the tree based on the number of leaves to keep
    if prune_by_num_leaves >= 0:
        custom_tree.prune_by_dist(prune_by_num_leaves, prevent_monochotomy)
        # After pruning the tree, update the dataset as well
        dataset.X = dataset.X[custom_tree.get_leaf_names()]
        assert all(custom_tree.get_leaf_names() == dataset.X.columns)

    print(dataset.y.value_counts())
    print(f"Number of leaves: {len(custom_tree.get_leaf_names())}")
        
    return custom_tree, dataset

def set_device(device):
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def taxonomy_to_tree(taxonomy_file, outfile):
    # Create an empty tree with a root node
    t = Tree(name='root')

    with open(taxonomy_file, 'r') as f:
        for line in f:
            microbe, taxonomy = line.strip().split('\t')
            taxon_levels = taxonomy.split('; ')

            # Create the tree by adding nodes iteratively
            parent = t
            for taxon in taxon_levels:
                taxon = taxon.replace('.', '')
                child = parent.search_nodes(name=taxon)
                if not child:
                    child = parent.add_child(name=taxon)
                else:
                    child = child[0]
                parent = child
            # Add the microbe as a leaf node
            parent.add_child(name=microbe)

    # Rename unnamed taxa to avoid conflicts
    id = 0
    for node in t.traverse():
        if not (node.is_leaf() or node.is_root()):
            if node.name.split('__')[1] == '': # Unnamed taxa, e.g. 'k__'
                node.name = f"{node.name}{id}"
                id += 1

    # Convert the tree to Newick format
    t.write(outfile=outfile, format=1, format_root_node=True)