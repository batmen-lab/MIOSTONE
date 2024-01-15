import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAccuracy, MulticlassAUROC,
                                         MulticlassAveragePrecision)


class ResultsAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results = None
        self.dataset = None
        self.target = None
        
    def _load_results(self, dataset, target, transfer_learning=False):
        self.dataset = dataset
        self.target = target

        results = []
        dir_name = 'predictions/'
        if transfer_learning:
            dir_name += 'transfer_learning/'
        results_dir = os.path.join(self.output_dir, dataset, target, dir_name)
        for filepath in glob.glob(os.path.join(results_dir, '*.json')):
            with open(filepath, 'r') as f:
                result = json.load(f)
                results.append(result)

        self.results = pd.DataFrame(results)
        for col in ['Model Hparams', 'Train Hparams', 'Mixup Hparams']:
            self.results[col] = self.results[col].apply(lambda x: frozenset(x.items()))
        for col in ['Epoch Val Labels', 'Epoch Val Logits']:
            self.results[col] = self.results[col].apply(lambda x: [np.array(y) for y in x])
        for col in ['Test Labels', 'Test Logits']:
            self.results[col] = self.results[col].apply(lambda x: np.array(x))

    def _compute_metircs(self):
        num_classes = self.results['Test Logits'].iloc[0].shape[1]
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassAUROC(num_classes=num_classes),
            MulticlassAveragePrecision(num_classes=num_classes),
        ])

         # Group the results by 'Model Type', 'Model Hparams', 'Train Hparams', 'Mixup Hparams'
        grouped_results = self.results.groupby(['Seed', 'Model Type', 'Model Hparams', 'Train Hparams', 'Mixup Hparams'])

        # Initialize dictionaries to store concatenated labels and logits
        concatenated_epoch_val_labels = {}
        concatenated_epoch_val_logits = {}
        concatenated_test_labels = {}
        concatenated_test_logits = {}

        # Concatenate the true labels and logits for each group
        for name, group in grouped_results:
            concatenated_epoch_val_labels[name] = np.hstack(group['Epoch Val Labels'].values)
            concatenated_epoch_val_logits[name] = np.hstack(group['Epoch Val Logits'].values)
            concatenated_test_labels[name] = np.concatenate(group['Test Labels'].values)
            concatenated_test_logits[name] = np.concatenate(group['Test Logits'].values)

        # Apply the metrics to the concatenated values
        rows = []
        for (seed, model_type, model_hparams, train_hparams, mixup_hparams), name in zip(grouped_results.groups.keys(), grouped_results.groups):
            epoch_val_scores = []
            for epoch in range(len(concatenated_epoch_val_labels[name])):
                epoch_val_scores.append(metrics(torch.tensor(concatenated_epoch_val_logits[name][epoch]), torch.tensor(concatenated_epoch_val_labels[name][epoch])))
            test_labels = torch.tensor(concatenated_test_labels[name])
            test_logits = torch.tensor(concatenated_test_logits[name])
            test_scores = metrics(test_logits, test_labels)
            row = {
                'Seed': seed,
                'Model Type': model_type,
                'epoch val Accuracy': [score['MulticlassAccuracy'].item() for score in epoch_val_scores],
                'epoch val AUROC': [score['MulticlassAUROC'].item() for score in epoch_val_scores],
                'epoch val AUPRC': [score['MulticlassAveragePrecision'].item() for score in epoch_val_scores],
                'test Accuracy': test_scores['MulticlassAccuracy'].item(),
                'test AUROC': test_scores['MulticlassAUROC'].item(),
                'test AUPRC': test_scores['MulticlassAveragePrecision'].item(),
            }
            for key, value in model_hparams:
                row[key] = value
            for key, value in train_hparams:
                row[key] = value
            for key, value in mixup_hparams:
                row[key] = value
            rows.append(row)
        
        # Create a dataframe from the rows
        self.scores = pd.DataFrame(rows)

    def _visualize_val_curves(self):
        # Flatten the scores list
        rows = []
        for i in range(len(self.scores)):
            original_row = self.scores.iloc[i]
            for epoch in range(len(original_row['epoch val Accuracy'])):
                row = {}
                row['Seed'] = original_row['Seed']
                row['Model Type'] = original_row['Model Type']
                row['epoch'] = epoch
                row['epoch val Accuracy'] = original_row['epoch val Accuracy'][epoch]
                row['epoch val AUROC'] = original_row['epoch val AUROC'][epoch]
                row['epoch val AUPRC'] = original_row['epoch val AUPRC'][epoch]
                rows.append(row)
        
        # Create a dataframe from the rows
        curves = pd.DataFrame(rows)

        # Set the style of the plots
        sns.set(style="whitegrid")

        # Define the metrics to plot
        metrics = ['Accuracy', 'AUROC', 'AUPRC']

        # Create a figure with subplots - one for each metric
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

        # Set the title of the figure
        fig.suptitle(f'{self.dataset} - {self.target}', fontsize=20)

        # Iterate over each metric to create a subplot
        for i, metric in enumerate(metrics):
            # Convert the scores list to a DataFrame
            # Use seaborn to create a line plot with error bars on the corresponding subplot
            sns.lineplot(data=curves,
                        x='epoch',
                        y='epoch val ' + metric,
                        hue='Model Type',
                        hue_order=['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone'],
                        errorbar="sd",
                        ax=axes[i])
          
            # Set subplot title and labels
            axes[i].set_title(f'{metric}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('')

        # Adjust layout for better visualization
        plt.tight_layout()

        # Show the plot
        plt.show()

    def _visualize_test_scores(self):
        # Set the style of the plots
        sns.set(style="whitegrid")

        # Define the metrics to plot
        metrics = ['test Accuracy', 'test AUROC', 'test AUPRC']

        # Create a figure with subplots - one for each metric
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

        # Set the title of the figure
        fig.suptitle(f'{self.dataset} - {self.target}', fontsize=20)

        # Iterate over each metric to create a subplot
        for i, metric in enumerate(metrics):
            # Use seaborn to create a bar plot with error bars on the corresponding subplot
            sns.barplot(data=self.scores,
                        x='max_epochs',
                        y=metric, 
                        hue='Model Type',
                        hue_order=['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone'],
                        errorbar="sd", 
                        errwidth=2, 
                        ax=axes[i])

            # Set subplot title and labels
            axes[i].set_title(f'{metric}')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')

            # Set the y-axis limit
            axes[i].set_ylim(0.48, 1.02)

            # Annotate the bars with the values
            self._annotate_bar_plot(axes[i])

        # Adjust layout for better visualization
        plt.tight_layout()

        # Show the plot
        plt.show()

    
    def _visualize_time_elapsed(self):
        # Set the style for the plots
        sns.set(style="whitegrid")

        # Create a plot
        plt.figure(figsize=(10, 6))

        # Use seaborn to create a bar plot
        sns.barplot(x='Model Type', 
                    y='Time Elapsed', 
                    data=self.results,
                    order=['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone'],
                    errorbar="sd", 
                    errwidth=2)

        # Set plot title and labels
        plt.title(f'{self.dataset} - {self.target}', fontsize=20)
        plt.xlabel('')
        plt.ylabel('Average Time Elapsed (seconds)')

        # Adjust layout for better visualization
        plt.tight_layout()

        # Show the plot
        plt.show()

    def _annotate_bar_plot(self, ax):
        # Set the offset for the annotations
        offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])

        # Iterate over each bar in the plot and annotate the height
        for p, err_bar in zip(ax.patches, ax.lines):
            bar_length = p.get_height()
            err_bar_length = err_bar.get_ydata()[1] - err_bar.get_ydata()[0]
            text_position = bar_length + err_bar_length / 2
            position = (p.get_x() + p.get_width() / 2., text_position + offset)
            ax.annotate(f"{bar_length:.2f}", xy=position, ha='center', va='center', fontsize=12, color='black')

       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to analyze')
    parser.add_argument('--target', type=str, required=True, help='The target to analyze')
    parser.add_argument('--visualize', type=str, required=True, choices=['scores', 'time', 'curves'], help='The type of visualization to create')
    parser.add_argument('--transfer_learning', action='store_true', help='Whether to analyze transfer learning results')
    args = parser.parse_args()

    analyzer = ResultsAnalyzer('../output/')
    analyzer._load_results(args.dataset, args.target, args.transfer_learning)
    analyzer._compute_metircs()
    if args.visualize == 'scores':
        analyzer._visualize_test_scores()
    elif args.visualize == 'time':
        analyzer._visualize_time_elapsed()
    elif args.visualize == 'curves':
        analyzer._visualize_val_curves()