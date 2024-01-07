import argparse
import json
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Timer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAccuracy, MulticlassAUROC,
                                         MulticlassAveragePrecision)

from data import MIOSTONEDataset, MIOSTONEMixup
from model import MLP, MIOSTONEModel, PopPhyCNN, TaxoNN
from pipeline import Pipeline


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
    def __init__(self, model, class_weight):
        super().__init__()
        self.model = model
        self.train_criterion = nn.CrossEntropyLoss(weight=class_weight)
        self.val_criterion = nn.CrossEntropyLoss()
        self.test_true_labels = []
        self.test_logits = []
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = self.train_criterion(logits, y)
        l0_reg = self.model.get_total_l0_reg()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_l0_reg', l0_reg, on_step=True, on_epoch=True, prog_bar=True)

        return loss + l0_reg
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = self.val_criterion(logits, y)
        l0_reg = self.model.get_total_l0_reg()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_l0_reg', l0_reg, on_step=True, on_epoch=True, prog_bar=True)

        return loss + l0_reg

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)

        # Store true labels and logits
        self.test_true_labels.append(y.detach().cpu())
        self.test_logits.append(logits.detach().cpu())

    def get_test_results(self):
        test_true_labels = torch.cat(self.test_true_labels, dim=0)
        test_logits = torch.cat(self.test_logits, dim=0)

        return test_true_labels, test_logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

class TrainingPipeline(Pipeline):
    def __init__(self, seed):
        super().__init__(seed)
        self.model_type = None
        self.model_hparams = {}
        self.train_hparams = {}
        self.mixup_hparams = {}

    def _create_subset(self, indices, normalize=True, one_hot_encoding=True, clr=True):
        X_subset = self.data.X[indices]
        y_subset = self.data.y[indices]
        subset = MIOSTONEDataset(X_subset, y_subset, self.data.features)
        if normalize:
            subset.normalize()
        if one_hot_encoding:
            subset.one_hot_encode()
        if clr:
            subset.clr_transform()
        return subset
    
    def _apply_mixup(self, train_dataset):
        mixup_processor = MIOSTONEMixup(train_dataset, self.tree)
        min_threshold = np.quantile(mixup_processor.distances[mixup_processor.distances != 0], self.mixup_hparams['q_interval'][0])
        max_threshold = np.quantile(mixup_processor.distances[mixup_processor.distances != 0], self.mixup_hparams['q_interval'][1])
        augmented_dataset = mixup_processor.mixup(min_threshold, max_threshold, self.mixup_hparams['num_samples'])
    
        return augmented_dataset

    def _create_classifier(self, train_dataset):
        in_features = train_dataset.X.shape[1]
        out_features = train_dataset.num_classes
        class_weight = train_dataset.class_weight if self.train_hparams['class_weight'] == 'balanced' else None

        if class_weight is None:
            class_weight = [1] * out_features
        if self.model_type == 'rf':
            class_weight = {key: value for key, value in enumerate(class_weight)}
            classifier = RandomForestClassifier(class_weight=class_weight, random_state=self.seed, **self.model_hparams)
        else:
            class_weight = torch.tensor(class_weight).float()
            if self.model_type == 'mlp':
                classifier = Classifier(MLP(in_features, out_features, **self.model_hparams), class_weight)
            elif self.model_type == 'taxonn':
                classifier = Classifier(TaxoNN(self.tree, out_features, train_dataset, **self.model_hparams), class_weight)
            elif self.model_type == 'popphycnn':
                classifier = Classifier(PopPhyCNN(self.tree, out_features, **self.model_hparams), class_weight)
            elif self.model_type == 'miostone':
                classifier = Classifier(MIOSTONEModel(self.tree, out_features, **self.model_hparams), class_weight)
            else:
                raise ValueError(f"Invalid model type: {self.model_type}")

        return classifier

    def _run_training(self, classifier, train_dataset, test_dataset):
        if self.model_type == 'rf':
            start_time = time.time()
            classifier.fit(train_dataset.X, train_dataset.y)
            time_elapsed = time.time() - start_time
            test_true_labels = torch.tensor(test_dataset.y)
            test_logits = torch.tensor(classifier.predict_proba(test_dataset.X))
        else:
            data_module = DataModule(train_dataset, test_dataset, test_dataset, batch_size=self.train_hparams['batch_size'], num_workers=1)
            timer = Timer()
            trainer = Trainer(max_epochs=self.train_hparams['max_epochs'],
                              enable_progress_bar=True, 
                              enable_model_summary=False,
                              enable_checkpointing=False,
                              log_every_n_steps=1,
                              callbacks=[timer],
                              accelerator='gpu')
            trainer.fit(classifier, datamodule=data_module)
            trainer.test(classifier, datamodule=data_module)
            test_true_labels, test_logits = classifier.get_test_results()
            time_elapsed = timer.time_elapsed('train')

        return test_true_labels, test_logits, time_elapsed
    
    def _save_model(self, classifier, filename):
        models_dir = self.output_dir + 'models/'
        if self.model_type == 'rf':
            pickle.dump(classifier, open(models_dir + filename + '.pkl', 'wb'))
        else:
            torch.save(classifier.model.state_dict(), models_dir + filename + '.pt')
    
    def _save_result(self, result, filename):
        results_dir = self.output_dir + 'results/'
        with open(results_dir + filename + '.json', 'w') as f:
            json.dump(result, f, indent=4)

    def _compute_metrics(self, metrics, test_logits, test_labels):
        metrics(test_logits, test_labels)
        metrics_result = metrics.compute()
        metrics.reset()
        return metrics_result

    def _compute_mean_std_cv_scores(self, metrics, overall_test_true_labels, overall_test_logits):
        metrics_per_fold = [self._compute_metrics(metrics, logits, labels)
                            for logits, labels in zip(overall_test_logits, overall_test_true_labels)]

        mean_metrics = np.mean([list(metrics.values()) for metrics in metrics_per_fold], axis=0)
        std_metrics = np.std([list(metrics.values()) for metrics in metrics_per_fold], axis=0)
        metrics_summary = {key: (mean, std) for key, mean, std in zip(metrics_per_fold[0].keys(), mean_metrics, std_metrics)}

        print("Mean CV Scores:")
        for metric, (mean, std) in metrics_summary.items():
            print(f"{metric}: Mean = {mean:.4f}, Std = {std:.4f}")

    def _compute_global_cv_scores(self, metrics, overall_test_true_labels, overall_test_logits):
        overall_test_logits = torch.cat(overall_test_logits, dim=0)
        overall_test_true_labels = torch.cat(overall_test_true_labels, dim=0)
        global_metrics = self._compute_metrics(metrics, overall_test_logits, overall_test_true_labels)
        
        print("Global CV Scores:")
        for metric, value in global_metrics.items():
            print(f"{metric}: {value:.4f}")
        

    def _train(self):
        # Check if data and tree are loaded
        if not self.data or not self.tree:
            raise ValueError('Please load data and tree first.')
        
        # Define metrics
        num_classes = len(np.unique(self.data.y))
        metrics = MetricCollection({
            'Accuracy': MulticlassAccuracy(num_classes=num_classes),
            'AUROC': MulticlassAUROC(num_classes=num_classes),
            'AUPRC': MulticlassAveragePrecision(num_classes=num_classes)
        })
        
        # Define cross-validation strategy
        skf = StratifiedKFold(n_splits=self.train_hparams['k_folds'], shuffle=True, random_state=self.seed)

        # Training loop
        overall_test_true_labels = []
        overall_test_logits = []
        for fold, (train_index, test_index) in enumerate(skf.split(self.data.X, self.data.y)):
            # Reset the seed for reproducibility
            seed_everything(self.seed)

            # Prepare datasets
            one_hot_encoding = True if self.mixup_hparams else False
            clr = False if self.model_type == 'popphycnn' else True
            train_dataset = self._create_subset(train_index, one_hot_encoding=one_hot_encoding, clr=clr)
            test_dataset = self._create_subset(test_index, one_hot_encoding=False, clr=clr)

            # Apply MIOSTONEMixup if specified
            if self.mixup_hparams:
                train_dataset = self._apply_mixup(train_dataset)

            # Create classifier 
            classifier = self._create_classifier(train_dataset)

            # Convert the datasets to tree matrices if necessary
            if self.model_type == 'popphycnn':
                scaler = train_dataset.to_tree_matrix(self.tree)
                test_dataset.to_tree_matrix(self.tree, scaler=scaler)

            # Run training
            test_true_labels, test_logits, time_elapsed = self._run_training(classifier, train_dataset, test_dataset)
            overall_test_true_labels.append(test_true_labels)
            overall_test_logits.append(test_logits)

            # Save test results
            result = {
                'Seed': self.seed,
                'Fold': fold,
                'Model Type': self.model_type,
                'Model Hparams': self.model_hparams,
                'Train Hparams': self.train_hparams,
                'Mixup Hparams': self.mixup_hparams,
                'True Labels': test_true_labels.numpy().tolist(), 
                'Logits': test_logits.numpy().tolist(),
                'Time Elapsed': time_elapsed,
            }

            # Save results and model
            filename = f"{self.seed}_{fold}_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._save_result(result, filename)
            self._save_model(classifier, filename)

        # Compute CV scores
        self._compute_mean_std_cv_scores(metrics, overall_test_true_labels, overall_test_logits)
        self._compute_global_cv_scores(metrics, overall_test_true_labels, overall_test_logits)

    def run(self, dataset, target, model_type):
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
        self.train_hparams['k_folds'] = 5
        self.train_hparams['batch_size'] = 512
        self.train_hparams['max_epochs'] = 100
        self.train_hparams['class_weight'] = 'balanced'

        # Configure default mixup parameters
        # self.mixup_hparams['num_samples'] = 100
        # self.mixup_hparams['q_interval'] = (0.0, 0.2)

        # Train the model
        self._train()

def run_training_pipeline(dataset, target, model_type, seed):
    pipeline = TrainingPipeline(seed=seed)
    pipeline.run(dataset, target, model_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument("--target", type=str, required=True, help="Target to predict.")
    parser.add_argument("--model_type", type=str, required=True, choices=['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone'], help="Model type to use.")
    args = parser.parse_args()

    run_training_pipeline(args.dataset, args.target, args.model_type, args.seed)