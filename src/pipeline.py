import json
import logging
import os
import pickle
from abc import ABC, abstractmethod

import torch
from ete4 import Tree
from lightning.fabric.utilities.seed import seed_everything

from data import MIOSTONEDataset, MIOSTONETree
from model import MIOSTONEModel
from baseline import MLP


class Pipeline(ABC):
    def __init__(self, seed):
        self.seed = seed
        self.data = None
        self.tree = None
        self.model = None

        # Set up logging
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        logging.getLogger("lightning.fabric").setLevel(logging.ERROR)

        # Set up seed
        seed_everything(self.seed)

    def _create_output_dir(self):
        subdirectories = ['models', 'results', 'embeddings', 'attributions', 'transfer_learning']
        for subdir in subdirectories:
            dir_path = os.path.join(self.output_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)

    def _validate_filepath(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError(f"File {filepath} does not exist.")

    def _load_data_and_tree(self, dataset, target, preprocess=True):
        # Define filepaths
        data_fp = f'../data/{dataset}/data.tsv.xz'
        meta_fp = f'../data/{dataset}/meta.tsv'
        target_fp = f'../data/{dataset}/{target}.py'
        tree_fp = '../data/WoL2/taxonomy.nwk'

        # Validate filepaths
        for fp in [data_fp, meta_fp, target_fp, tree_fp]:
            self._validate_filepath(fp)

        # Load data
        X, y, features = MIOSTONEDataset.preprocess(data_fp, meta_fp, target_fp)
        self.data = MIOSTONEDataset(X, y, features)

        # Load tree
        ete_tree = Tree(open(tree_fp), parser=1)
        ete_tree.name = 'root'

        # Convert the tree to a MIOSTONETree
        self.tree = MIOSTONETree(ete_tree)

        # Prune the tree to only include the taxa in the dataset
        self.tree.prune(self.data.features)

        # Compute the depth of each node in the tree
        self.tree.compute_depths()

        # Compute the index of each node in the tree
        self.tree.compute_indices()

        # Order the features in the dataset according to the tree
        self.data.order_features_by_tree(self.tree)

        # Create output directory if it does not exist
        self.output_dir = f'../output/{dataset}/{target}/'
        self._create_output_dir()

        # Preprocess the dataset
        if preprocess:
            self.data.normalize()
            self.data.clr_transform()

    def _load_model(self, model_fp, results_fp):
        # Validate filepaths
        for fp in [model_fp, results_fp]:
            self._validate_filepath(fp)

        # Load model hyperparameters
        with open(results_fp) as f:
            results = json.load(f)
            model_type = results['Model Type']
            model_hparams = results['Model Hparams']
        
        # Load model
        in_features = self.data.X.shape[1]
        out_features = self.data.num_classes
        if model_type == 'rf':
            self.model = pickle.load(open(model_fp, 'rb'))
        else:
            if model_type == 'mlp':
                self.model = MLP(in_features, out_features, **model_hparams)
            elif model_type == 'miostone':
                self.model = MIOSTONEModel(tree=self.tree, out_features=out_features, **model_hparams)
            else:
                raise ValueError(f"Invalid model type: {model_type}")
        
            self.model.load_state_dict(torch.load(model_fp))

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
        
