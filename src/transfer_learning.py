import argparse
import copy
import os
from datetime import datetime

import numpy as np
from lightning.pytorch import seed_everything
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAccuracy, MulticlassAUROC,
                                         MulticlassAveragePrecision)

from data import MIOSTONEDataset
from train import TrainingPipeline


class TransferLearningPipeline(TrainingPipeline):
    def __init__(self, seed):
        super().__init__(seed)
        self.pretrain_data = None

    def _create_output_subdir(self):
        super()._create_output_subdir()
        self.pred_dir = self.pred_dir + 'transfer_learning/'
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)

    def _load_pretain_data(self, dataset, target, drop_features=True):
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
        if drop_features:
            self.pretrain_data.drop_features_by_tree(self.tree)

        # Add features that are leaves in the tree
        self.pretrain_data.add_features_by_tree(self.tree)

        # Order features by tree
        self.pretrain_data.order_features_by_tree(self.tree)

    def _create_classifier(self, train_dataset, metrics):
        classifier = super()._create_classifier(train_dataset, metrics)
        if self.model is not None:
            print("Loading pretrained model...")
            if self.model_type == 'rf':
                classifier = copy.deepcopy(self.model)
            else:
                classifier.model.load_state_dict(self.model.state_dict())
                if self.model_type == 'miostone':
                    for layer in classifier.model.hidden_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

        return classifier

    def _pretrain(self):
        # Check if data and tree are loaded
        if not self.pretrain_data or not self.tree:
            raise RuntimeError("Data and tree must be loaded before training.")
        
        # Set up seed
        seed_everything(0, workers=True)

        # Prepare datasets
        clr = False if self.model_type == 'popphycnn' else True
        train_index = np.arange(len(self.pretrain_data))
        train_dataset = self._create_subset(self.pretrain_data, train_index, one_hot_encoding=False, clr=clr)

        # Define metrics
        num_classes = len(np.unique(self.data.y))
        metrics = MetricCollection({
            'Accuracy': MulticlassAccuracy(num_classes=num_classes),
            'AUROC': MulticlassAUROC(num_classes=num_classes),
            'AUPRC': MulticlassAveragePrecision(num_classes=num_classes)
        })

        # Create classifier 
        classifier = self._create_classifier(train_dataset, metrics)

        # Convert to tree matrix if specified and apply standardization
        if self.model_type == 'popphycnn':
            train_dataset.to_tree_matrix(self.tree)
        else:
            train_dataset.standardize()

        # Run training
        filename = f"{self.seed}_pretrain_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_training(classifier, train_dataset, train_dataset, train_dataset, filename)

        # Set the model to the trained model
        self.model = classifier if self.model_type == 'rf' else classifier.model

        # Reset the seed
        seed_everything(self.seed, workers=True)

    def run(self, dataset, pretrain_dataset, target, model_type, num_epochs, *args, **kwargs):
        # Load data and tree
        self._load_data_and_tree(dataset, target, preprocess=False)

        # Create output directory
        self._create_output_subdir()

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
        self.train_hparams['k_folds'] = 5
        self.train_hparams['batch_size'] = 512
        self.train_hparams['max_epochs'] = 30
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

        # Configure default fine-tuning parameters
        self.train_hparams['max_epochs'] = num_epochs
        self.train_hparams['class_weight'] = 'balanced'
        self.train_hparams['num_frozen_layers'] = 0

        # Run training
        self._train()

def run_transfer_learning_pipeline(dataset, pretrain_dataset, target, model_type, num_epochs, seed):
    pipeline = TransferLearningPipeline(seed)
    pipeline.run(dataset, pretrain_dataset, target, model_type, num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use for fine-tuning.')
    parser.add_argument('--pretrain_dataset', type=str, required=True, help='Dataset to use for pretraining.')
    parser.add_argument('--target', type=str, required=True, help='Target to predict.')
    parser.add_argument("--model_type", type=str, required=True, choices=['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone'], help="Model type to use.")
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs to fine-tune for.')
    args = parser.parse_args()

    run_transfer_learning_pipeline(args.dataset, args.pretrain_dataset, args.target, args.model_type, args.num_epochs, args.seed)