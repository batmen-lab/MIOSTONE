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

    def _load_pretain_data(self, data_fp, meta_fp, target_fp, drop_features=True):
        # Validate filepaths
        for fp in [data_fp, meta_fp, target_fp]:
            self._validate_filepath(fp)

        # Load data
        self.pretrain_data = MIOSTONEDataset.init_from_files(data_fp, meta_fp, target_fp)

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
            if self.model_type in ['rf', 'lr', 'svm']:
                classifier = copy.deepcopy(self.model)
            else:
                classifier.model.load_state_dict(self.model.state_dict())
                if self.model_type == 'miostone':
                    for layer in classifier.model.hidden_layers[:self.train_hparams['num_frozen_layers']]:
                        for param in layer.parameters():
                            param.requires_grad = False

        return classifier

    def _pretrain(self):
        # Check if data and tree are loaded
        if self.pretrain_data is None or self.tree is None:
            raise RuntimeError("Data and tree must be loaded before training.")
        
        # Set up seed
        seed_everything(0, workers=True)

        # Prepare datasets
        normalize = False
        clr = False if self.model_type == 'popphycnn' else True
        train_index = np.arange(len(self.pretrain_data))
        train_dataset = self._create_subset(self.pretrain_data, 
                                            train_index, 
                                            normalize=normalize,
                                            one_hot_encoding=False, 
                                            clr=clr)

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
            train_dataset.to_popphycnn_matrix(self.tree)

        # Run training
        filename = f"{self.seed}_pretrain_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_training(classifier, train_dataset, train_dataset, train_dataset, filename)

        # Set the model to the trained model
        self.model = classifier if self.model_type in ['rf', 'lr', 'svm'] else classifier.model

        # Reset the seed
        seed_everything(self.seed, workers=True)

    def run(self, dataset, pretrain_dataset, target, model_type, num_epochs, num_frozen_layers, *args, **kwargs):
        # Define filepaths
        self.filepaths = {
            'data': f'../data/{dataset}/data.tsv.xz',
            'meta': f'../data/{dataset}/meta.tsv',
            'target': f'../data/{dataset}/{target}.py',
            'tree': '../data/WoL2/taxonomy.nwk'
        }

        # Load tree
        self._load_tree(self.filepaths['tree'])
        
        # Load data
        self._load_data(self.filepaths['data'], self.filepaths['meta'], self.filepaths['target'], preprocess=False)

        # Create output directory
        self._create_output_subdir()

        # Configure default model parameters
        self.model_type = model_type
        if self.model_type == 'miostone':
            self.model_hparams['node_min_dim'] = 1
            self.model_hparams['node_dim_func'] = 'linear'
            self.model_hparams['node_dim_func_param'] = 0.6
            self.model_hparams['node_gate_type'] = 'concrete'
            self.model_hparams['node_gate_param'] = 0.3
            self.model_hparams['prune_mode'] = 'taxonomy'
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
        self.train_hparams['max_epochs'] = num_epochs
        self.train_hparams['class_weight'] = 'balanced'
        self.train_hparams['pretrain_num_epochs'] = num_epochs
            
        # Pretrain the model if necessary
        if pretrain_dataset != dataset:
            # Define filepaths
            self.filepaths['pretrain_data'] = f'../data/{pretrain_dataset}/data.tsv.xz'
            self.filepaths['pretrain_meta'] = f'../data/{pretrain_dataset}/meta.tsv'
            self.filepaths['pretrain_target'] = f'../data/{pretrain_dataset}/{target}.py'
            
            # Load pretrain data
            self._load_pretain_data(self.filepaths['pretrain_data'], self.filepaths['pretrain_meta'], self.filepaths['pretrain_target'])

            # Pretrain the model
            self._pretrain()

        # Configure default fine-tuning parameters
        self.train_hparams['max_epochs'] = 200
        self.train_hparams['class_weight'] = 'balanced'
        if self.model_type == 'miostone':
            self.train_hparams['num_frozen_layers'] = num_frozen_layers

        # Run training
        self._train()

def run_transfer_learning_pipeline(dataset, pretrain_dataset, target, model_type, seed, num_epochs=25, num_frozen_layers=7):
    pipeline = TransferLearningPipeline(seed)
    pipeline.run(dataset, pretrain_dataset, target, model_type, num_epochs, num_frozen_layers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use for fine-tuning.')
    parser.add_argument('--pretrain_dataset', type=str, required=True, help='Dataset to use for pretraining.')
    parser.add_argument('--target', type=str, required=True, help='Target to predict.')
    parser.add_argument("--model_type", type=str, required=True, choices=['rf', 'mlp', 'svm', 'popphycnn', 'taxonn', 'miostone'], help="Model type to use.")
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to pretrain for.')
    parser.add_argument('--num_frozen_layers', type=int, default=7, help='Number of layers to freeze.')
    args = parser.parse_args()

    run_transfer_learning_pipeline(args.dataset, args.pretrain_dataset, args.target, args.model_type, args.num_epochs, args.num_frozen_layers, args.seed)