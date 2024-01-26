import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy import stats
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAccuracy, MulticlassAUROC,
                                         MulticlassAveragePrecision)


class ResultsAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results = None
        self.dataset = None
        self.target = None
        self.results = None
        self.scores = None
        self.model_type_labels = {
            'taxonn': 'TaxoNN',
            'popphycnn': 'PopPhy-CNN',
            'mlp': 'Multilayer Perceptron',
            'svm': 'Support Vector Machine',
            'rf': 'Random Forest',
            'miostone': 'MIOSTONE'
        }
        self.model_type_palette = {
            'taxonn': '#aec7e8',       # Lightest blue for TaxonNN
            'popphycnn': '#7eaedc',    # Slightly darker blue for PopPhyCNN
            'mlp': '#5da5da',          # Medium blue shade for MLP
            'svm': '#3498db',          # Darker blue for SVM
            'rf': '#1f77b4',           # Darkest blue for Random Forest
            'miostone': '#ffd700'      # Gold for MIOSTONE
        }
        self.dataset_labels = {
            'rumc_pd': 'RUMC - PD',
            'alzbiom_ad': 'AlzBiom - AD',
            'ibd200_type': 'IBD200 - Type',
            'asd_stage': 'ASD - Stage',
            'tbc_pd': 'TBC - PD',
            'hmp2_type': 'HMP2 - Type',
            'gd_cohort': 'GD - Cohort'
        }
        
    def _load_results(self, dataset=None, target=None, transfer_learning=False):
        self.dataset = dataset
        self.target = target

        results = []
        if self.dataset is not None and self.target is not None:
            if os.path.exists(os.path.join(self.output_dir, dataset, target, 'results.csv')):
                self.results = pd.read_csv(os.path.join(self.output_dir, dataset, target, 'results.csv'))
                return
            dir_name = 'predictions/'
            if transfer_learning:
                dir_name += 'transfer_learning/'
            results_dir = os.path.join(self.output_dir, dataset, target, dir_name)
            filepaths = glob.glob(os.path.join(results_dir, '*.json'))
        else:
            if os.path.exists(os.path.join(self.output_dir, 'results.csv')):
                self.results = pd.read_csv(os.path.join(self.output_dir, 'results.csv'))
                return
            filepaths = []
            for dataset in os.listdir(self.output_dir):
                if os.path.isdir(os.path.join(self.output_dir, dataset)):
                    for target in os.listdir(os.path.join(self.output_dir, dataset)):
                        filepaths.extend(glob.glob(os.path.join(self.output_dir, dataset, target, 'predictions/*.json')))

        for filepath in filepaths:
            with open(filepath, 'r') as f:
                result = json.load(f)
                if self.dataset is None and self.target is None:
                    dataset = filepath.split('/')[-4]
                    target = filepath.split('/')[-3]
                    result['dataset_target'] = dataset + '_' + target
                results.append(result)
        
        self.results = pd.DataFrame(results)
        for col in ['Model Hparams', 'Train Hparams', 'Mixup Hparams']:
            self.results[col] = self.results[col].apply(lambda x: frozenset(x.items()))
        for col in ['Epoch Val Labels', 'Epoch Val Logits']:
            self.results[col] = self.results[col].apply(lambda x: [np.array(y) for y in x] if isinstance(x, list) else x)
        for col in ['Test Labels', 'Test Logits']:
            self.results[col] = self.results[col].apply(lambda x: np.array(x))

        # Save the results to a file
        if self.dataset is not None and self.target is not None:
            self.results.to_csv(f'{self.output_dir}/{self.dataset}/{self.target}/results.csv', index=False)
        else:
            self.results.to_csv(f'{self.output_dir}/results.csv', index=False)

    def _compute_metrics(self):
        if self.dataset is not None and self.target is not None:
            if os.path.exists(os.path.join(self.output_dir, self.dataset, self.target, 'scores.csv')):
                self.scores = pd.read_csv(os.path.join(self.output_dir, self.dataset, self.target, 'scores.csv'))
                return
        else:
            if os.path.exists(os.path.join(self.output_dir, 'scores.csv')):
                self.scores = pd.read_csv(os.path.join(self.output_dir, 'scores.csv'))
                return
        
        num_classes = self.results['Test Logits'].iloc[0].shape[1]
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassAUROC(num_classes=num_classes),
            MulticlassAveragePrecision(num_classes=num_classes),
        ])

        # Group the results by the specified columns
        group_cols = ['Seed', 'Model Type', 'Model Hparams', 'Train Hparams', 'Mixup Hparams']
        if self.dataset is None and self.target is None:
            group_cols += ['dataset_target']
        grouped_results = self.results.groupby(group_cols)

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

        for cols, name in zip(grouped_results.groups.keys(), grouped_results.groups):
            model_type = cols[1]
            model_hparams = cols[2]
            train_hparams = cols[3]
            mixup_hparams = cols[4]

            test_labels = torch.tensor(concatenated_test_labels[name])
            test_logits = torch.tensor(concatenated_test_logits[name])
            test_scores = metrics(test_logits, test_labels)
            row = {
                 #'Fold': fold,
                'Seed': cols[0],
                'Model Type': model_type,
                'test Accuracy': test_scores['MulticlassAccuracy'].item(),
                'test AUROC': test_scores['MulticlassAUROC'].item(),
                'test AUPRC': test_scores['MulticlassAveragePrecision'].item(),
            }
            if self.dataset is None and self.target is None:
                row['dataset_target'] = cols[5]

            if model_type not in ['rf', 'lr', 'svm']:
                epoch_val_scores = []
                for epoch in range(len(concatenated_epoch_val_labels[name])):
                    epoch_val_scores.append(metrics(torch.tensor(concatenated_epoch_val_logits[name][epoch]), torch.tensor(concatenated_epoch_val_labels[name][epoch])))
                row['epoch val Accuracy'] = [score['MulticlassAccuracy'].item() for score in epoch_val_scores]
                row['epoch val AUROC'] = [score['MulticlassAUROC'].item() for score in epoch_val_scores]
                row['epoch val AUPRC'] = [score['MulticlassAveragePrecision'].item() for score in epoch_val_scores]

            for key, value in model_hparams:
                row[key] = value
            for key, value in train_hparams:
                row[key] = value
            for key, value in mixup_hparams:
                row[key] = value

            if 'node_gate_type' in row:
                row['node_gate_type'] = row['node_gate_type'] + '_' + str(row['node_gate_param'])
            
            if 'pretrain_num_epochs' not in row:
                row['transfer_learning'] = 'Training from Scratch'
                row['pretrain_num_epochs'] = 0
            elif row['max_epochs'] == 0:
                row['transfer_learning'] = 'Zero-Shot'
            else:
                row['transfer_learning'] = 'Fine-tuning'

            if 'percent_features' not in row:
                row['percent_features'] = 1.0

            rows.append(row)
        
        # Create a dataframe from the rows
        self.scores = pd.DataFrame(rows)
        
        # Save the scores to a file
        if self.dataset is not None and self.target is not None:
            self.scores.to_csv(f'{self.output_dir}/{self.dataset}/{self.target}/scores.csv', index=False)
        else:
            self.scores.to_csv(f'{self.output_dir}/scores.csv', index=False)

    def _visualize_val_curves(self):
        # Flatten the scores list
        rows = []
        for i in range(len(self.scores)):
            original_row = self.scores.iloc[i]
            if original_row['Model Type'] != 'miostone':
                continue
            for epoch in range(len(original_row['epoch val Accuracy'])):
                row = original_row.copy()
                row['epoch'] = epoch
                row['epoch val Accuracy'] = original_row['epoch val Accuracy'][epoch]
                row['epoch val AUROC'] = original_row['epoch val AUROC'][epoch]
                row['epoch val AUPRC'] = original_row['epoch val AUPRC'][epoch]
                rows.append(row)
        
        # Create a dataframe from the rows
        curves = pd.DataFrame(rows)
        curves = curves[curves["Model Type"] == "miostone"]
        curves = curves[curves["transfer_learning"] != "Zero-Shot"]
       
        # Set the style of the plots
        sns.set(style='white')

        # Define the metrics to plot
        metrics = ['AUROC', 'AUPRC']

        # Iterate over each metric
        for metric in metrics:
            # Create a plot
            plt.figure(figsize=(10, 10))
            # Convert the scores list to a DataFrame
            # Use seaborn to create a line plot with error bars on the corresponding subplot
            sns.lineplot(data=curves,
                        x='epoch',
                        y='epoch val ' + metric,
                        hue='transfer_learning',
                        # hue=curves[['transfer_learning', 'pretrain_num_epochs']].apply(tuple, axis=1),
                        # hue='Model Type',
                        # hue_order=['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone'],
                        palette=['#1f77b4', '#ffd700'],
                        errorbar="ci")
          
            # Set plot title and labels
            plt.title(f'{metric}')
            plt.xlabel('Epoch')
            plt.ylabel('')

            # Set the y-axis limit
            plt.ylim(0.45, 0.85)

            # Get the scores for the two conditions and sort them by epoch and seed
            condition1_scores = curves[curves["transfer_learning"] == "Training from Scratch"].sort_values(by=['epoch', 'Seed'])["epoch val " + metric].values
            condition2_scores = curves[curves["transfer_learning"] == "Fine-tuning"].sort_values(by=['epoch', 'Seed'])["epoch val " + metric].values

            # Perform a Wilcoxon signed-rank test to determine whether the difference between the two conditions is significant
            statistic, p_value = stats.wilcoxon(condition1_scores, condition2_scores, alternative='less', method='exact')

            # Annotate the subplot with the p-value
            '''
            axes[i].text(0.5, 0.1, f'p-value: {p_value}', 
                         horizontalalignment='center', 
                        verticalalignment='center', 
                        transform=axes[i].transAxes, 
                        fontsize=12)
            '''

            # Adjusting legend
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # Adjust layout for better visualization
            plt.tight_layout()

            # Show the plot
            plt.show()
    
    def _visualize_test_scores(self):
        # self.scores = self.scores[self.scores['Model Type'] == 'miostone']
        self.scores = self.scores[(self.scores['prune_mode'] == 'taxonomy') | (self.scores['Model Type'] != 'miostone')]
        self.scores = self.scores[(self.scores['node_gate_type'] == 'concrete_0.3') | (self.scores['Model Type'] != 'miostone')]
        self.scores = self.scores[(self.scores['node_dim_func'] == 'linear') | (self.scores['Model Type'] != 'miostone')]
        self.scores = self.scores[(self.scores['percent_features'] == 1.0) | (self.scores['Model Type'] != 'miostone')]
        # self.scores["percent_features"] = self.scores["percent_features"].apply(lambda x: str(int(x * 100)) + "%")
        # Set the style of the plots
        sns.set_theme(style="white")

        # Define the metrics to plot
        metrics = ['test AUROC', 'test AUPRC']

        # Iterate over each metric
        for metric in metrics:
            # Print the best seed for miostone for each dataset
            # print(self.scores[self.scores['Model Type'] == 'miostone'].groupby(['dataset_target']).apply(lambda x: x.loc[x[metric].idxmax()][['Seed', metric]]))

            # Create a plot
            plt.figure(figsize=(20, 10))

            # Use seaborn to create a bar plot with error bars on the corresponding subplot
            sns.barplot(data=self.scores,
                        # x = 'transfer_learning',
                        # order=['Zero-Shot', 'Training from Scratch', 'Fine-tuning'],
                        # x = "Model Type",
                        # order=['rf', 'svm', 'mlp', 'popphycnn', 'taxonn', 'miostone'],
                        # x= "dataset_target",
                        # order=['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'hmp2_type', 'gd_cohort'],
                        # order=['alzbiom_ncbi_ad', 'asd_ncbi_stage'],
                        y=metric, 
                        # hue="node_gate_type",
                        # hue_order=['deterministic_1.0', 'concrete_0.3'],
                        # hue="node_dim_func",
                        # hue_order=['const', 'linear'],
                        # hue = "percent_features",
                        # hue_order=['1%', '5%', '10%', '20%', '50%', '100%'],
                        hue ="Model Type",
                        hue_order=['rf', 'svm', 'mlp', 'popphycnn', 'taxonn', 'miostone'],
                        # hue = "transfer_learning",
                        # hue_order=['Zero-Shot', 'Training from Scratch', 'Fine-tuning'],
                        palette=self.model_type_palette,
                        # palette=['#1f77b4', '#3498db', '#5da5da', '#7eaedc', '#aec7e8', '#ffd700'],
                        errorbar="ci",
                        errwidth=2)

            # Set lot title and labels
            plt.title(f'{metric.replace("test ", "")}')
            plt.xlabel('')
            plt.ylabel('')

            # Set x-axis tick labels
            # plt.xticks([0, 1, 2, 3, 4, 5, 6], ['RUMC - PD', 'AlzBiom - AD', 'IBD200 - Type', 'ASD - Stage', 'TBC - PD', 'HMP2 - Type', 'GD - Cohort'])
            # plt.xticks([0, 1], ['AlzBiom - AD', 'ASD - Stage'])
            # plt.xticks([0, 1, 2, 3, 4, 5], ['Random Forest', 'Support Vector Machine', 'Multilayer Perceptron', 'PopPhy-CNN', 'TaxoNN', 'MIOSTONE'])

            # Set the y-axis limit
            # plt.ylim(0.48, 0.82)
            plt.ylim(0.45, 1.05)
            
            # Annotate the bars with the values
            self._annotate_bar_plot(plt.gca())

            # Rename the legend labels and move the legend outside the plot
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles, [self.model_type_labels[label] for label in labels], loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.legend(handles, ['Non-linear representation only', 'Adaptive representation'], loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.gca().legend(handles, ['Fixed hidden dimension', 'Adaptive hidden dimension'], loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # Adjust layout for better visualization
            plt.tight_layout()

            # Show the plot
            plt.show()

    def _visualize_time_elapsed(self):
        # Set the style for the plots
        sns.set(style="white")

        # Create a plot
        plt.figure(figsize=(20, 5))

        # Use seaborn to create a bar plot
        sns.barplot(x='dataset_target',
                    order=['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'hmp2_type', 'gd_cohort'],
                    y='Time Elapsed', 
                    data=self.results,
                    hue='Model Type',
                    hue_order=['rf', 'svm', 'mlp', 'popphycnn', 'taxonn', 'miostone'],
                    palette=self.model_type_palette,
                    errorbar="sd", 
                    errwidth=2)
        
        # Set log scale for the y-axis
        plt.yscale('log')
    
        # Set plot title and labels
        plt.title(f'Computational Cost')
        plt.xlabel('')
        plt.ylabel('Average Training Time (Seconds per Fold)')

        # Set x-axis tick labels
        plt.xticks([0, 1, 2, 3, 4, 5, 6], ['RUMC - PD', 'AlzBiom - AD', 'IBD200 - Type', 'ASD - Stage', 'TBC - PD', 'HMP2 - Type', 'GD - Cohort'])

        # Annotate the bars with the values
        self._annotate_bar_plot(plt.gca())

        # Rename the legend labels and move the legend outside the plot
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.gca().legend(handles, [self.model_type_labels[label] for label in labels], loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Adjust layout for better visualization
        plt.tight_layout()

        # Show the plot
        plt.show()

    def _annotate_bar_plot(self, ax):
        is_log_scale = ax.get_yscale() == 'log'

        # Get the limits of the y-axis
        ylim = ax.get_ylim()
        log_range = np.log10(ylim[1]/ylim[0]) if is_log_scale else None

        # Iterate over each bar in the plot and annotate the height
        for p, err_bar in zip(ax.patches, ax.lines):
            bar_length = p.get_height()
            err_bar_length = err_bar.get_ydata()[1] - err_bar.get_ydata()[0]
            text_position = bar_length + err_bar_length / 2

            if is_log_scale:
                # Calculate an offset that is proportional to the log scale
                log_offset = 0.01 * log_range
                y = text_position * (10 ** log_offset)
            else:
                # Use a fixed offset for linear scale
                offset = 0.01 * (ylim[1] - ylim[0])
                y = text_position + offset

            x = p.get_x() + p.get_width() / 2
            position = (x, y)
            ax.annotate(f"{bar_length:.2f}", xy=position, ha='center', va='center', fontsize=10, color='black')


    def _visualize_percent_features(self):
        self.scores = self.scores[self.scores['Model Type'] == 'miostone']
        self.scores = self.scores[self.scores['prune_mode'] == 'taxonomy']
        self.scores = self.scores[self.scores['node_gate_type'] == 'concrete_0.3']
        self.scores = self.scores[self.scores['node_dim_func'] == 'linear']
        self.scores["percent_features"] = self.scores["percent_features"].apply(lambda x: str(int(x * 100)) + "%")
        # Set the style of the plots
        sns.set_theme(style="white")

        # Define the metrics to plot
        metrics = ['test AUROC', 'test AUPRC']

        # Iterate over each metric
        for metric in metrics:
            plt.figure(figsize=(10, 10))
            # Use seaborn to create a catplot with error bars
            sns.pointplot(data=self.scores,
                        x="percent_features",
                        order=['1%', '5%', '10%', '20%', '50%', '100%'],
                        y=metric, 
                        hue="dataset_target",
                        hue_order=reversed(['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'hmp2_type', 'gd_cohort']),
                        errorbar="ci")

            # Set plot title and labels
            plt.title(f'{metric.replace("test ", "")}')
            plt.xlabel('Top Features (%) Used')
            plt.ylabel('')

            # Set the y-axis limit
            plt.ylim(0.45, 1.05)

             # Rename the legend labels and move the legend outside the plot
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.gca().legend(handles, [self.dataset_labels[label] for label in labels], loc='center left', bbox_to_anchor=(1, 0.5))

            # Adjust layout for better visualization
            plt.tight_layout()

            # Show the plot
            plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='The dataset to analyze')
    parser.add_argument('--target', type=str, help='The target to analyze')
    parser.add_argument('--visualize', type=str, required=True, choices=['scores', 'time', 'curves', 'percent_features'], help='The type of visualization to generate')
    parser.add_argument('--transfer_learning', action='store_true', help='Whether to analyze transfer learning results')
    args = parser.parse_args()

    analyzer = ResultsAnalyzer('../output/')
    analyzer._load_results(args.dataset, args.target, args.transfer_learning)
    analyzer._compute_metrics()
    if args.visualize == 'scores':
        analyzer._visualize_test_scores()
    elif args.visualize == 'time':
        analyzer._visualize_time_elapsed()
    elif args.visualize == 'curves':
        analyzer._visualize_val_curves()
    elif args.visualize == 'percent_features':
        analyzer._visualize_percent_features()