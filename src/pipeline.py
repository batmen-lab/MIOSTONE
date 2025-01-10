import json
import logging
import os
from abc import ABC, abstractmethod

import scanpy as sc
import torch
from lightning.pytorch import seed_everything
from scanpy import AnnData

from baseline import MLP, PopPhyCNN, TaxoNN, MDeep, DeepBiome, PhCNN
from data import MIOSTONEDataset, MIOSTONETree
from model import MIOSTONEModel


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

    def _validate_filepath(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError(f"File {filepath} does not exist.")
        
    def _load_tree(self, tree_fp):
        # Validate filepath
        self._validate_filepath(tree_fp)

        # Load tree from file
        if tree_fp.endswith('.nwk'):
            self.tree = MIOSTONETree.init_from_nwk(tree_fp)
        elif tree_fp.endswith('.tsv'):
            self.tree = MIOSTONETree.init_from_tsv(tree_fp)
        else:
            raise ValueError(f"Invalid tree filepath: {tree_fp}")

    def _load_data(self, data_fp, meta_fp, target_fp, percent_features=1.0, prune=True, preprocess=True):
        # Validate filepaths
        for fp in [data_fp, meta_fp, target_fp]:
            self._validate_filepath(fp)

        # Load data
        self.data = MIOSTONEDataset.init_from_files(data_fp, meta_fp, target_fp)

        # Create output directory if it does not exist
        dataset_name = data_fp.split('/')[-2]
        target_name = target_fp.split('/')[-1].split('.')[0]
        self.output_dir = f'../output/{dataset_name}/{target_name}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        if percent_features < 1.0:
            # Reduce number of features using highly_variable_genes
            # Convert to AnnData format for scanpy compatibility
            adata = AnnData(self.data.X.copy())
            adata.var_names = self.data.features

            # Run highly variable genes detection
            top_features = int(percent_features * self.data.X.shape[1])
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=top_features)
            # Extract the indices of the highly variable genes
            hvg_indices = adata.var['highly_variable'].to_numpy().nonzero()[0]

            # Subset the original dataset to only include these genes
            self.data.X = self.data.X[:, hvg_indices]
            self.data.features = self.data.features[hvg_indices]

        # Prune the tree to only include the taxa in the dataset
        if prune:
            self.tree.prune(self.data.features)
        else:
            self.data.add_features_by_tree(self.tree)

        # Compute the depth of each node in the tree
        self.tree.compute_depths()

        # Compute the index of each node in the tree
        self.tree.compute_indices()

        # Order the features in the dataset according to the tree
        self.data.order_features_by_tree(self.tree)

        # Preprocess the dataset
        if preprocess:
            self.data.clr_transform()


    def _load_model(self, model_fp, results_fp):
        # Validate filepaths
        for fp in [model_fp, results_fp]:
            self._validate_filepath(fp)

        # Load model hyperparameters
        with open(results_fp) as f:
            results = json.load(f)
            self.model_type = results['Model Type']
            model_hparams = results['Model Hparams']
        
        # Load model
        in_features = self.data.X.shape[1]
        out_features = self.data.num_classes
        if self.model_type == 'taxonn':
            self.model = torch.load(model_fp)
        else:
            if self.model_type == 'miostone':
                self.model = MIOSTONEModel(self.tree, out_features, **model_hparams)
            elif self.model_type == 'mlp':
                self.model = MLP(in_features, out_features, **model_hparams)
            elif self.model_type == 'popphycnn':
                self.model = PopPhyCNN(self.tree, out_features, **model_hparams)
            elif self.model_type == 'mdeep':
                self.model = MDeep(self.tree, out_features, **model_hparams)
            elif self.model_type == 'deepbiome':
                self.model = DeepBiome(self.tree, out_features, **model_hparams)
            elif self.model_type == 'phcnn':
                self.model = PhCNN(self.tree, out_features, **model_hparams)
            else:
                raise ValueError(f"Invalid model type: {self.model_type}")
            
            self.model.load_state_dict(torch.load(model_fp))

    @abstractmethod
    def _create_output_subdir(self):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
        
