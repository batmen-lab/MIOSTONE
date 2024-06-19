import argparse
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAUROC,
                                         MulticlassAveragePrecision)

from baseline import MLP, PopPhyCNN, TaxoNN
from data import MIOSTONEDataset
from model import MIOSTONEModel
from pipeline import Pipeline
from treenn import TreeNN


class DataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size, num_workers):  
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
class Classifier(LightningModule):
    def __init__(self, model, class_weight, metrics):
        super().__init__()
        self.model = model
        self.train_criterion = nn.CrossEntropyLoss(weight=class_weight)
        self.val_criterion = nn.CrossEntropyLoss()
        self.metrics = metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Initialize lists to store logits and labels
        self.epoch_val_logits = []
        self.epoch_val_labels = []
        self.test_logits = None
        self.test_labels = None
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = self.train_criterion(logits, y)
        l0_reg = self.model.get_total_l0_reg()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
    
        return loss + l0_reg

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = self.val_criterion(logits, y)
        l0_reg = self.model.get_total_l0_reg()
        self.validation_step_outputs.append({'logits': logits, 'labels': y})
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False)

        # Calculate metrics
        self.metrics.to(logits.device)
        scores = self.metrics(logits, y)
        for key, value in scores.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        return loss + l0_reg
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        logits = torch.cat([x['logits'] for x in outputs], dim=0)
        labels = torch.cat([x['labels'] for x in outputs], dim=0)

        # Store logits and labels
        self.epoch_val_logits.append(logits.detach().cpu().numpy().tolist())
        self.epoch_val_labels.append(labels.detach().cpu().numpy().tolist())

        # Reset validation step outputs
        self.validation_step_outputs = []

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        self.test_step_outputs.append({'logits': logits, 'labels': y})
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        logits = torch.cat([x['logits'] for x in outputs], dim=0)
        labels = torch.cat([x['labels'] for x in outputs], dim=0)

        # Store logits and labels
        self.test_logits = logits.detach().cpu().numpy().tolist()
        self.test_labels = labels.detach().cpu().numpy().tolist()

        # Reset test step outputs
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

class TrainingPipeline(Pipeline):
    def __init__(self, seed):
        super().__init__(seed)
        self.model_type = None
        self.model_hparams = {}
        self.train_hparams = {}
        self.mixup_hparams = {}
        self.output_subdir = None

    def _create_output_subdir(self):
        self.pred_dir = self.output_dir + 'predictions/'
        self.model_dir = self.output_dir + 'models/'
        for dir in [self.pred_dir, self.model_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def _create_subset(self, data, indices, normalize=True, one_hot_encoding=True, clr=True):
        X_subset = data.X[indices]
        y_subset = data.y[indices]
        ids_subset = data.ids[indices]
        subset = MIOSTONEDataset(X_subset, y_subset, ids_subset, data.features)
        if normalize:
            subset.normalize()
        if one_hot_encoding:
            subset.one_hot_encode()
        if clr:
            subset.clr_transform()
        return subset

    def _create_classifier(self, train_dataset, metrics):
        in_features = train_dataset.X.shape[1]
        out_features = train_dataset.num_classes
        class_weight = train_dataset.class_weight if self.train_hparams['class_weight'] == 'balanced' else [1] * out_features

        if self.model_type in ['rf', 'svm']:
            class_weight = {key: value for key, value in enumerate(class_weight)}
            if self.model_type == 'rf':
                classifier = RandomForestClassifier(class_weight=class_weight, random_state=self.seed)
            elif self.model_type == 'svm':
                classifier = SVC(kernel='linear', probability=True, class_weight=class_weight, random_state=self.seed)
        else:
            class_weight = torch.tensor(class_weight).float()
            if self.model_type == 'mlp':
                model = MLP(in_features, out_features, **self.model_hparams)
            elif self.model_type == 'taxonn':
                model = TaxoNN(self.tree, out_features, train_dataset, **self.model_hparams)
            elif self.model_type == 'popphycnn':
                model = PopPhyCNN(self.tree, out_features, **self.model_hparams)
            elif self.model_type == 'miostone':
                model = MIOSTONEModel(self.tree, out_features, **self.model_hparams)
            elif self.model_type == 'treenn':
                model = TreeNN(device='cpu', tree=self.tree, **self.model_hparams)
            else:
                raise ValueError(f"Invalid model type: {self.model_type}")
    
            classifier = Classifier(model, class_weight, metrics)

        return classifier

    def _run_sklearn_training(self, classifier, train_dataset, test_dataset):
        time_elapsed = 0
        if self.train_hparams['max_epochs'] > 0:
            start_time = time.time()
            classifier.fit(train_dataset.X, train_dataset.y)
            time_elapsed = time.time() - start_time

        test_labels = test_dataset.y.tolist()
        test_logits = classifier.predict_proba(test_dataset.X).tolist()

        return {
            'test_labels': test_labels,
            'test_logits': test_logits,
            'time_elapsed': time_elapsed,
        }

    def _run_pytorch_training(self, classifier, train_dataset, val_dataset, test_dataset, filename):
        timer = Timer()
        logger = TensorBoardLogger(self.output_dir + 'logs/', name=filename)
        data_module = DataModule(train_dataset, val_dataset, test_dataset, batch_size=self.train_hparams['batch_size'], num_workers=1)

        trainer = Trainer(
            max_epochs=self.train_hparams['max_epochs'],
            enable_progress_bar=True, 
            enable_model_summary=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            logger=logger,
            callbacks=[timer],
            accelerator='gpu',
            devices=1,
            deterministic=True,
        )
        trainer.fit(classifier, datamodule=data_module)
        trainer.test(classifier, datamodule=data_module)

        return {
            'epoch_val_labels': classifier.epoch_val_labels,
            'epoch_val_logits': classifier.epoch_val_logits,
            'test_labels': classifier.test_labels,
            'test_logits': classifier.test_logits,
            'time_elapsed': timer.time_elapsed('train')
        }

    def _run_training(self, classifier, train_dataset, val_dataset, test_dataset, filename):
        if self.model_type in ['rf', 'svm']:
            return self._run_sklearn_training(classifier, train_dataset, test_dataset)
        else:
            return self._run_pytorch_training(classifier, train_dataset, val_dataset, test_dataset, filename)
    
    def _save_model(self, classifier, save_dir, filename):
        if self.model_type in ['rf', 'svm']:
            pickle.dump(classifier, open(save_dir + filename + '.pkl', 'wb'))
        elif self.model_type == 'taxonn':
            torch.save(classifier.model, save_dir + filename + '.pt')
        else:
            torch.save(classifier.model.state_dict(), save_dir + filename + '.pt')
    
    def _save_result(self, result, save_dir, filename):
        with open(save_dir + filename + '.json', 'w') as f:
            result_to_save = {
                'Seed': self.seed,
                'Fold': filename.split('_')[1],
                'Model Type': self.model_type,
                'Model Hparams': self.model_hparams,
                'Train Hparams': self.train_hparams,
                'Mixup Hparams': self.mixup_hparams,
                'Test Labels': result['test_labels'],
                'Test Logits': result['test_logits'],
                'Time Elapsed': result['time_elapsed']
            }
            if self.model_type not in ['rf', 'svm']:
                result_to_save['Epoch Val Labels'] = result['epoch_val_labels']
                result_to_save['Epoch Val Logits'] = result['epoch_val_logits']
            json.dump(result_to_save, f, indent=4)

    def _train(self):
        # Check if data and tree are loaded
        if not self.data or not self.tree:
            raise ValueError('Please load data and tree first.')
        
        # Define metrics
        num_classes = len(np.unique(self.data.y))
        metrics = MetricCollection({
            'AUROC': MulticlassAUROC(num_classes=num_classes),
            'AUPRC': MulticlassAveragePrecision(num_classes=num_classes)
        })
        
        # Define cross-validation strategy
        kf = KFold(n_splits=self.train_hparams['k_folds'], shuffle=True, random_state=self.seed)

        # Training loop
        fold_test_labels = []
        fold_test_logits = []
        for fold, (train_index, test_index) in enumerate(kf.split(self.data.X, self.data.y)):
            # Prepare datasets
            normalize = True if self.model_type == 'popphycnn' else False
            clr = False if self.model_type == 'popphycnn' else True
            train_dataset = self._create_subset(self.data, 
                                                train_index, 
                                                normalize=normalize,
                                                one_hot_encoding=False, 
                                                clr=clr)
            test_dataset = self._create_subset(self.data, 
                                               test_index, 
                                               normalize=normalize,
                                               one_hot_encoding=False, 
                                               clr=clr)

            # Create classifier 
            classifier = self._create_classifier(train_dataset, metrics)

            # Convert to tree matrix if specified and apply standardization
            if self.model_type == 'popphycnn':
                scaler = train_dataset.to_popphycnn_matrix(self.tree)
                test_dataset.to_popphycnn_matrix(self.tree, scaler=scaler)

            # Set filename
            filename = f"{self.seed}_{fold}_{self.model_type}"
            if self.model_type == 'miostone':
                filename += f"_{self.model_hparams['node_gate_type']}{self.model_hparams['node_gate_param']}"
                filename += f"_{self.model_hparams['node_dim_func']}{self.model_hparams['node_dim_func_param']}"
                filename += f"_{self.model_hparams['prune_mode']}"
            if "percent_features" in self.train_hparams:
                filename += f"_p{self.train_hparams['percent_features']}"
            if "num_frozen_layers" in self.train_hparams:
                filename += f"_frozen{self.train_hparams['num_frozen_layers']}"
            if "pretrain_num_epochs" in self.train_hparams:
                filename += f"_pretrain{self.train_hparams['pretrain_num_epochs']}"
                if self.train_hparams['max_epochs'] == 0:
                    filename += "_zs"
            filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Run training
            result = self._run_training(classifier, train_dataset, test_dataset, test_dataset, filename)
            fold_test_labels.append(torch.tensor(result['test_labels']))
            fold_test_logits.append(torch.tensor(result['test_logits']))
                     
            # Save results and model
            self._save_result(result, self.pred_dir, filename)
            self._save_model(classifier, self.model_dir, filename)

        # Calculate metrics
        test_labels = torch.cat(fold_test_labels, dim=0)
        test_logits = torch.cat(fold_test_logits, dim=0)
        metrics.to(test_labels.device)
        test_scores = metrics(test_logits, test_labels)
        print(f"Test scores:")
        for key, value in test_scores.items():
            print(f"{key}: {value.item()}")

    def run(self, dataset, target, model_type, *args, **kwargs):
        # Define filepaths
        self.filepaths = {
            'data': f'../data/{dataset}/data.tsv.xz',
            'meta': f'../data/{dataset}/meta.tsv',
            'target': f'../data/{dataset}/{target}.py',
            'tree': '../data/WoL2/taxonomy.nwk' if not dataset.endswith('_ncbi') else '../data/WoL2/taxonomy_ncbi.nwk'
            # 'tree': '../data/WoL2/phylogeny.nwk'
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
        elif self.model_type == 'treenn':
            self.model_hparams['node_min_dim'] = 1
            self.model_hparams['node_dim_func'] = 'linear'
            self.model_hparams['node_dim_func_param'] = 0.6
            self.model_hparams['node_gate_type'] = 'concrete'
            self.model_hparams['node_gate_param'] = 0.3
            self.model_hparams['output_dim'] = 2
            self.model_hparams['output_depth'] = 0
            self.model_hparams['output_truncation'] = False

        # Configure default training parameters
        self.train_hparams['k_folds'] = 5
        self.train_hparams['batch_size'] = 512
        self.train_hparams['max_epochs'] = 200
        self.train_hparams['class_weight'] = 'balanced'
        # self.train_hparams['percent_features'] = kwargs.get('percent_features', 1)
        
        # Train the model
        self._train()

def run_training_pipeline(dataset, target, model_type, seed, *args, **kwargs):
    pipeline = TrainingPipeline(seed=seed)
    pipeline.run(dataset, target, model_type, *args, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument("--target", type=str, required=True, help="Target to predict.")
    parser.add_argument("--model_type", type=str, required=True, choices=['rf', 'svm', 'mlp', 'taxonn', 'popphycnn', 'miostone', 'treenn'], help="Model type to use.")
    # parser.add_argument("--percent_features", type=float, default=1, help="Percentage of features to use.")
    args = parser.parse_args()

    run_training_pipeline(args.dataset, args.target, args.model_type, args.seed)