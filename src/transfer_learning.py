import argparse
import json

import numpy as np

from data import MIOSTONEDataset
from train import TrainingPipeline


class TransferLearningPipeline(TrainingPipeline):
    def __init__(self, seed):
        super().__init__(seed)
        self.pretrain_data = None

    def _load_pretain_data(self, dataset, target):
        # Define filepaths
        data_fp = f'../data/{dataset}/data.tsv.xz'
        meta_fp = f'../data/{dataset}/meta.tsv'
        target_fp = f'../data/{dataset}/{target}.py'

        # Validate filepaths
        for fp in [data_fp, meta_fp, target_fp]:
            self._validate_filepath(fp)

        # Load data
        X, y, features = MIOSTONEDataset.preprocess(data_fp, meta_fp, target_fp)
        self.pretrain_data = MIOSTONEDataset(X, y, features)

        # Drop features that are not leaves in the tree
        self.pretrain_data.drop_features_by_tree(self.tree)

        # Add features that are leaves in the tree
        self.pretrain_data.add_features_by_tree(self.tree)

        # Order features by tree
        self.pretrain_data.order_features_by_tree(self.tree)

    def _pretrain(self):
         # Check if data and tree are loaded
        if not self.pretrain_data or not self.tree:
            raise RuntimeError("Data and tree must be loaded before training.")

        # Prepare datasets
        clr = False if self.model_type == 'popphycnn' else True
        train_index = np.arange(len(self.pretrain_data))
        train_dataset = self._create_subset(self.pretrain_data, train_index, one_hot_encoding=False, clr=clr)

        # Create classifier 
        classifier = self._create_classifier(train_dataset)

        # Convert the datasets to tree matrices if necessary
        if self.model_type == 'popphycnn':
            train_dataset.to_tree_matrix(self.tree)

        # Run training
        self._run_training(classifier, train_dataset, train_dataset)

        # Set the model to the trained model
        self.model = classifier if self.model_type == 'rf' else classifier.model
        
    def _save_result(self, result, filename):
        results_dir = self.output_dir + 'transfer_learning/'
        with open(results_dir + filename + '.json', 'w') as f:
            json.dump(result, f, indent=4)

    def run(self, dataset, pretrain_dataset, target, model_type, tl_epochs, *args, **kwargs):
        # Load data and tree
        self._load_data_and_tree(dataset, target, preprocess=False)

        # Configure default model parameters
        self.model_type = model_type
        if self.model_type == 'miostone':
            self.model_hparams['node_min_dim'] = 1
            self.model_hparams['node_dim_func'] = 'linear'
            self.model_hparams['node_dim_func_param'] = 0.7
            self.model_hparams['node_gate_type'] = 'concrete'
            self.model_hparams['node_gate_param'] = 0.3
        elif self.model_type == 'popphycnn':
            self.model_hparams['num_kernel'] = 32
            self.model_hparams['kernel_height'] = 3
            self.model_hparams['kernel_width'] = 10
            self.model_hparams['num_fc_nodes'] = 512
            self.model_hparams['num_cnn_layers'] = 1
            self.model_hparams['num_fc_layers'] = 1
            self.model_hparams['dropout'] = 0.3

        # Configure default training parameters
        self.train_hparams['batch_size'] = 512
        self.train_hparams['max_epochs'] = 200
        self.train_hparams['class_weight'] = 'balanced'

        # Configure default mixup parameters
        # self.mixup_hparams['num_samples'] = 100
        # self.mixup_hparams['q_interval'] = (0.0, 0.2)

        # Pretrain the model if necessary
        if pretrain_dataset != dataset:
             # Load pretrain data
            self._load_pretain_data(pretrain_dataset, target)

            # Pretrain the model
            self._pretrain()

        # Configure fine-tuning/zero-shot learning parameters
        self.train_hparams['k_folds'] = 5
        self.train_hparams['max_epochs'] = tl_epochs

        # Run training
        self._train()

def run_transfer_learning_pipeline(dataset, pretrain_dataset, target, model_type, tl_epochs, seed):
    pipeline = TransferLearningPipeline(seed)
    pipeline.run(dataset, pretrain_dataset, target, model_type, tl_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use for fine-tuning.')
    parser.add_argument('--pretrain_dataset', type=str, required=True, help='Dataset to use for pretraining.')
    parser.add_argument('--target', type=str, required=True, help='Target to predict.')
    parser.add_argument("--model_type", type=str, required=True, choices=['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone'], help="Model type to use.")
    parser.add_argument('--tl_epochs', type=int, required=True, help='Number of epochs to fine-tune. 0 for zero-shot learning.')
    args = parser.parse_args()

    run_transfer_learning_pipeline(args.dataset, args.pretrain_dataset, args.target, args.model_type, args.tl_epochs, args.seed)