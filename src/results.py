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
        
    def _load_results(self, dataset, target):
        self.dataset = dataset
        self.target = target

        results = []
        results_dir = os.path.join(self.output_dir, dataset, target, 'results')
        for filepath in glob.glob(os.path.join(results_dir, '*.json')):
            with open(filepath, 'r') as f:
                result = json.load(f)
                results.append(result)

        self.results = pd.DataFrame(results)
        for col in ['Model Hparams', 'Train Hparams', 'Mixup Hparams']:
            self.results[col] = self.results[col].apply(lambda x: frozenset(x.items()))
        for col in ['True Labels', 'Logits']:
            self.results[col] = self.results[col].apply(lambda x: np.array(x))

    def _compute_metircs(self):
        num_classes = self.results['Logits'].iloc[0].shape[1]
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassAUROC(num_classes=num_classes),
            MulticlassAveragePrecision(num_classes=num_classes),
        ])
         # Group the results by 'Model Type', 'Model Hparams', 'Train Hparams', 'Mixup Hparams'
        grouped_results = self.results.groupby(['Seed', 'Model Type', 'Model Hparams', 'Train Hparams', 'Mixup Hparams'])

        # Initialize dictionaries to store concatenated labels and logits
        concatenated_labels = {}
        concatenated_logits = {}

        # Concatenate the true labels and logits for each group
        for name, group in grouped_results:
            concatenated_labels[name] = np.concatenate(group['True Labels'].values)
            concatenated_logits[name] = np.concatenate(group['Logits'].values)

        # Apply the metrics to the concatenated values
        rows = []
        for (seed, model_type, model_hparams, train_hparams, mixup_hparams), name in zip(grouped_results.groups.keys(), grouped_results.groups):
            true_labels = torch.tensor(concatenated_labels[name])
            logits = torch.tensor(concatenated_logits[name])
            scores = metrics(logits, true_labels)
            row = {
                'Seed': seed,
                'Model Type': model_type,
                'Model Hparams': model_hparams,
                'Train Hparams': train_hparams,
                'Mixup Hparams': mixup_hparams,
                'Accuracy': scores['MulticlassAccuracy'].item(),
                'AUROC': scores['MulticlassAUROC'].item(),
                'AUPRC': scores['MulticlassAveragePrecision'].item(),
            }
            rows.append(row)
        
        # Create a dataframe from the rows
        self.scores = pd.DataFrame(rows)

    def _visualize_scores(self):
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
            # Use seaborn to create a bar plot with error bars on the corresponding subplot
            sns.barplot(x='Model Type', 
                        y=metric, 
                        data=self.scores, 
                        order=['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone'],
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
            ax.annotate(f"{bar_length:.2f}", xy=position, ha='center', va='center', fontsize=16, color='black')

       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to analyze')
    parser.add_argument('--target', type=str, required=True, help='The target to analyze')
    parser.add_argument('--visualize', type=str, required=True, choices=['scores', 'time'], help='The type of visualization to generate')
    args = parser.parse_args()

    analyzer = ResultsAnalyzer('../output/')
    analyzer._load_results(args.dataset, args.target)
    analyzer._compute_metircs()
    if args.visualize == 'scores':
        analyzer._visualize_scores()
    elif args.visualize == 'time':
        analyzer._visualize_time_elapsed()