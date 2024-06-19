import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy import stats
# from statannotations.Annotator import Annotator
# from statannotations.stats.StatTest import StatTest
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAccuracy, MulticlassAUROC,
                                         MulticlassAveragePrecision)
from tqdm import tqdm


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
            print(f'Loading results for {dataset} - {target}...')
            dir_name = 'predictions/'
            if transfer_learning:
                dir_name += 'transfer_learning/'
            results_dir = os.path.join(self.output_dir, dataset, target, dir_name)
            filepaths = glob.glob(os.path.join(results_dir, '*.json'))
        else:
            print('Loading all results...')
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

    def _compute_metrics(self):
        print('Computing metrics...')
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

        for cols, name in tqdm(zip(grouped_results.groups.keys(), grouped_results.groups), total=len(grouped_results.groups.keys())):
            seed = cols[0]
            model_type = cols[1]
            model_hparams = cols[2]
            train_hparams = cols[3]
            mixup_hparams = cols[4]

            test_labels = torch.tensor(concatenated_test_labels[name])
            test_logits = torch.tensor(concatenated_test_logits[name])
            test_scores = metrics(test_logits, test_labels)
            row = {
                # 'Fold': fold,
                'Seed': seed,
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
            if original_row['Model Type'] != 'taxonn':
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
        curves = curves[curves["Model Type"] == "taxonn"]
        curves = curves[curves["transfer_learning"] != "Zero-Shot"]
       
        # Set the style of the plots
        sns.set(style='white')

        # Define the metrics to plot
        metrics = ['AUROC', 'AUPRC']

        # Iterate over each metric
        for metric in metrics:
            # Create a plot
            plt.figure(figsize=(10, 8))
            # Convert the scores list to a DataFrame
            # Use seaborn to create a line plot with error bars on the corresponding subplot
            sns.lineplot(data=curves,
                        x='epoch',
                        y='epoch val ' + metric,
                        hue='transfer_learning',
                        # hue=curves[['transfer_learning', 'pretrain_num_epochs']].apply(tuple, axis=1),
                        # hue='Model Type',
                        # hue_order=['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone'],
                        # palette=['#1f77b4', '#ffd700'],
                        errorbar='ci')
          
            # Set plot title and labels
            plt.title(f'{metric}')
            plt.xlabel('Epoch')
            plt.ylabel('')

            # Set the y-axis limit
            plt.ylim(0.45, 0.85)

            # Get the scores for the last epoch for each condition
            condition1_scores = curves[(curves["transfer_learning"] == "Training from Scratch") & (curves["epoch"] == 200)][f'epoch val {metric}'].values
            print(len(condition1_scores))
            condition2_scores = curves[(curves["transfer_learning"] == "Fine-tuning") & (curves["epoch"] == 200)][f'epoch val {metric}'].values
            print(len(condition2_scores))

            # Perform a t-test for the two conditions for the last epoch
            statistic, p_value = stats.ttest_rel(condition2_scores, condition1_scores, alternative='greater')
            print(f'p-value for {metric}: {p_value}')

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
        self.scores = self.scores[(self.scores['Model Type'] == 'miostone') | (self.scores['Model Type'] == 'treenn')]
        # self.scores = self.scores[self.scores["Model Type"] != 'treenn']
        # self.scores["percent_features"] = self.scores["percent_features"].apply(lambda x: str(int(x * 100)) + "%")
        # Set the style of the plots
        sns.set_theme(style="white")

        # Define the metrics to plot
        metrics = ['test AUROC', 'test AUPRC']

        # Define hatch patterns for each transfer learning type
        hatch_patterns = {'Zero-Shot': '//', 'Training from Scratch': '--', 'Fine-tuning': '..'}

        # Iterate over each metric
        for metric in metrics:
            # Print the best seed for miostone for each dataset
            # print(self.scores[self.scores['Model Type'] == 'miostone'].groupby(['dataset_target']).apply(lambda x: x.loc[x[metric].idxmax()][['Seed', metric]]))

            # If a dataset and target are specified, filter the scores for that dataset and target, print the best seed for miostone for that dataset and target (no dataset_target column in the scores dataframe)
            if self.dataset is not None and self.target is not None:
                print(self.scores.loc[self.scores[self.scores['Model Type'] == 'miostone'][metric].idxmax()][['Seed', metric]])
            plt.figure(figsize=(20, 10))

            # Use seaborn to create a bar plot with error bars on the corresponding subplot
            ax = sns.barplot(data=self.scores,
                        # x = 'transfer_learning',
                        # order=['Zero-Shot', 'Training from Scratch', 'Fine-tuning'],
                        # x= "Model Type",
                        # order=['rf', 'svm', 'mlp', 'popphycnn', 'taxonn', 'miostone'],
                        x= "dataset_target",
                        order=['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'gd_cohort', 'hmp2_type'],
                        # order=['alzbiom_ncbi_ad', 'asd_ncbi_stage'],
                        y=metric, 
                        hue="Model Type",
                        # hue="node_gate_type",
                        # hue_order=['deterministic_1.0', 'concrete_0.3'],
                        # hue="node_dim_func",
                        # hue_order=['const', 'linear'],
                        # hue = "percent_features",
                        # hue_order=['1%', '5%', '10%', '20%', '50%', '100%'],
                        # hue ="transfer_learning",
                        hue_order = ["treenn", "miostone"],
                        # hue_order=['rf', 'svm', 'mlp', 'popphycnn', 'taxonn', 'miostone'],
                        # hue = "transfer_learning",
                        # hue_order=['Zero-Shot', 'Training from Scratch', 'Fine-tuning'],
                        # palette=self.model_type_palette,
                        # palette=['#1f77b4', '#3498db', '#5da5da', '#7eaedc', '#aec7e8', '#ffd700'],
                        palette=['#1f77b4', '#ffd700'],
                        # errorbar="ci",
                        ci=95,
                        errwidth=2)

            # Set lot title and labels
            plt.title(f'{metric.replace("test ", "")}')
            plt.xlabel('')
            plt.ylabel('')

            # Set x-axis tick labels
            plt.xticks([0, 1, 2, 3, 4, 5, 6], ['RUMC - PD', 'AlzBiom - AD', 'IBD200 - Type', 'ASD - Stage', 'TBC - PD', 'GD - Cohort', 'HMP2 - Type'])
            # plt.xticks([0, 1], ['AlzBiom - AD', 'ASD - Stage'])
            # plt.xticks([0, 1], ['MIOSTONE-Phylogeny', 'MIOSTONE-Taxonomy'])
            # plt.xticks([0, 1, 2, 3, 4, 5], ['Random Forest', 'SVM', 'MLP', 'PopPhy-CNN', 'TaxoNN', 'MIOSTONE'], fontsize=12)

            # Set the y-axis limit
            # plt.ylim(0.45, self.scores[metric].max() + 0.05)
            plt.ylim(0.45, 1.05)
            
            # Annotate the bars with the values
            self._annotate_bar_plot(plt.gca())

            # Remove the lengend
            # ax.get_legend().remove()


            # Apply hatches to each bar
            '''
            for i, bar in enumerate(ax.patches):
                # Determine the transfer learning type based on the bar's hue_index
                hue_index = i % len(hatch_patterns)
                transfer_type = ['Zero-Shot', 'Training from Scratch', 'Fine-tuning'][hue_index]
                bar.set_hatch(hatch_patterns[transfer_type])
            '''

            # Rename the legend labels and move the legend outside the plot
            # handles, labels = plt.gca().get_legend_handles_labels()
            # plt.legend(handles, [self.model_type_labels[label] for label in labels], loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.legend(handles, ['Deterministic representation aggregation', 'Data-driven representation aggregation'])
            # plt.gca().legend(handles, ['Fixed neuron dimensionality', 'Taxonomy-dependent dimensionality'])
            # plt.legend(handles, ['MIOSTONE w/ phylogenetic tree', 'MIOSTONE w/ taxonomic tree'])
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            # Perform statistical tests and add annotations
            def ttest_rel_less(group_data1, group_data2, **stats_params):
                return stats.ttest_rel(group_data1, group_data2, alternative='less', **stats_params)
            
            custom_long_name = 't-test paired samples (less)'
            custom_short_name = 't-test_rel_greater'
            custom_func = ttest_rel_less
            custom_stat_test = StatTest(custom_func, custom_long_name, custom_short_name)
            
            x = 'dataset_target'
            order = ['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'gd_cohort', 'hmp2_type']
            # hue = 'node_gate_type'
            # hue_order = ['deterministic_1.0', 'concrete_0.3']
            # hue = 'node_dim_func'
            # hue_order = ['const', 'linear']
            hue = 'Model Type'
            hue_order = ['treenn', 'miostone']
            # hue_order = ['rf', 'svm', 'mlp', 'popphycnn', 'taxonn', 'miostone']

            # pairs = [((dataset, 'const'), (dataset, 'linear')) for dataset in ['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'gd_cohort', 'hmp2_type']]
            # pairs = [((dataset, 'deterministic_1.0'), (dataset, 'concrete_0.3')) for dataset in ['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'gd_cohort', 'hmp2_type']]
            # pairs = [((dataset, model_type), (dataset, 'miostone')) for dataset in ['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'hmp2_type', 'gd_cohort'] for model_type in ['rf', 'svm', 'mlp', 'popphycnn', 'taxonn']]
            # pairs = [('rf', 'miostone'), ('svm', 'miostone'), ('mlp', 'miostone'), ('popphycnn', 'miostone'), ('taxonn', 'miostone')]
            # pairs = [((dataset, 'rf'), (dataset, 'miostone')) for dataset in ['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'gd_cohort', 'hmp2_type']]
            pairs = [((dataset, 'treenn'), (dataset, 'miostone')) for dataset in ['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'gd_cohort', 'hmp2_type']]
    
            # x = 'transfer_learning'
            # order = ['Zero-Shot', 'Training from Scratch', 'Fine-tuning']
            # pairs = [('Training from Scratch', 'Fine-tuning')]
            annotator = Annotator(ax, pairs, data=self.scores, x=x, y=metric, order=order, hue=hue, hue_order=hue_order)
            annotator.configure(test=custom_stat_test, text_format='full', show_test_name=False)
            annotator.apply_and_annotate()

            # Adjust layout for better visualization
            plt.tight_layout()

            # Show the plot
            plt.show()

    def _visualize_test_scores_dot_plot(self):
        self.scores = self.scores[self.scores['Model Type'] == 'taxonn']
        self.scores = self.scores[(self.scores['prune_mode'] == 'taxonomy') | (self.scores['Model Type'] != 'miostone')]
        self.scores = self.scores[(self.scores['node_gate_type'] == 'concrete_0.3') | (self.scores['Model Type'] != 'miostone')]
        self.scores = self.scores[(self.scores['node_dim_func'] == 'linear') | (self.scores['Model Type'] != 'miostone')]
        self.scores = self.scores[(self.scores['percent_features'] == 1.0) | (self.scores['Model Type'] != 'miostone')] 

        # Set the style of the plots
        sns.set_theme(style="white")

        # Define the metrics to plot
        metrics = ['test AUROC', 'test AUPRC']

        # Iterate over each metric
        for metric in metrics:
            # Prepare a DataFrame to hold pairs of MIOSTONE and other model scores for the same seed
            comparison_df = pd.DataFrame()
            
            # For each unique seed, get MIOSTONE score and compare with each other model's score
            for seed in self.scores['Seed'].unique():
                '''
                miostone_score = self.scores[(self.scores['Seed'] == seed) & (self.scores['Model Type'] == 'miostone')][metric].values
                other_scores = self.scores[(self.scores['Seed'] == seed) & (self.scores['Model Type'] != 'miostone')][['Model Type', metric]]
                
                # If MIOSTONE score exists for this seed
                if len(miostone_score) > 0:
                    # For each other model's score, create a row with both scores
                    for _, row in other_scores.iterrows():
                        comparison_df = comparison_df._append({
                            'Seed': seed,
                            'Model Type': row['Model Type'],
                            'MIOSTONE Score': miostone_score[0],
                            f'{metric} Score': row[metric]
                        }, ignore_index=True)
                '''
                fine_tuning_score = self.scores[(self.scores['Seed'] == seed) & (self.scores['transfer_learning'] == 'Fine-tuning')][metric].values
                tranning_from_scratch_score = self.scores[(self.scores['Seed'] == seed) & (self.scores['transfer_learning'] == 'Training from Scratch')]

                # If fine-tuning score exists for this seed
                if len(fine_tuning_score) > 0:
                    # For each other model's score, create a row with both scores
                    for _, row in tranning_from_scratch_score.iterrows():
                        comparison_df = comparison_df._append({
                            'Seed': seed,
                            'Model Type': row['Model Type'],
                            'fine-tuning Score': fine_tuning_score[0],
                            f'{metric} Score': row[metric]
                        }, ignore_index=True)


            # Perform statistical tests
            # self.perform_statistical_tests(comparison_df, metric)
            
            # Now plot the comparison for this metric
            plt.figure(figsize=(10, 10))
            '''
            sns.scatterplot(data=comparison_df, 
                            x=f'{metric} Score', 
                            y='MIOSTONE Score', 
                            hue='Model Type', 
                            hue_order=['rf', 'svm', 'mlp', 'popphycnn', 'taxonn'],
                            style='Model Type', 
                            s=100)
            '''
            sns.scatterplot(data=comparison_df,
                            x=f'{metric} Score',
                            y='fine-tuning Score',
                            # hue='Model Type',
                            # palette=self.model_type_palette,
                            s=100)
            
            # Add plot labels and title
            '''
            plt.xlabel(f'Other Models Score')
            plt.ylabel('MIOSTONE Score')
            plt.title(f'{self.dataset} - {self.target} ({metric})')
            '''
            plt.xlabel('Training from Scratch Score')
            plt.ylabel('Fine-tuning Score')
            plt.title(f'{metric}'.replace('test ', ''))


            # Set the x and y axis limits
            '''
            min_score = comparison_df[[f'{metric} Score', 'MIOSTONE Score']].min().min() - 0.05
            max_score = comparison_df[[f'{metric} Score', 'MIOSTONE Score']].max().max() + 0.05
            '''
            min_score = comparison_df[[f'{metric} Score', 'fine-tuning Score']].min().min() - 0.05
            max_score = comparison_df[[f'{metric} Score', 'fine-tuning Score']].max().max() + 0.05
            min_score = 0.45
            max_score = 0.85
            plt.xlim(min_score, max_score)
            plt.ylim(min_score, max_score)

            # Adding a reference line to indicate where the scores are equal
            plt.plot([min_score, max_score], [min_score, max_score], color='gray', linestyle='--')
            
            # Enhance legend
            # handles, labels = plt.gca().get_legend_handles_labels()
            # plt.legend(handles, [self.model_type_labels[label] for label in labels], loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.legend(handles, ['Zero-Shot', 'Training from Scratch'], loc='center left', bbox_to_anchor=(1, 0.5))

            # Remove the legend
            plt.gca().legend().remove()

            # Adjust layout and display the plot
            plt.tight_layout()
            plt.show()

    def perform_statistical_tests(self, comparison_df, metric_name):
        # Unique model types for comparison
        model_types = comparison_df['Model Type'].unique()
        
        # Initialize a list to store summary results
        summary_results = []

        for model_type in model_types:
            # Scores for MIOSTONE vs. the current model
            miostone_scores = comparison_df[comparison_df['Model Type'] == model_type].sort_values(by='Seed')['MIOSTONE Score'].values
            model_scores = comparison_df[comparison_df['Model Type'] == model_type].sort_values(by='Seed')[f'{metric_name} Score'].values
            
            # Check if both arrays have the same length and non-zero length
            if len(miostone_scores) == len(model_scores) and len(miostone_scores) > 0:
                # Perform the Wilcoxon signed-rank test
                stat, p_value = stats.ttest_rel(miostone_scores, model_scores, alternative='greater')
                
                # Determine the result based on the original p-value
                significant = p_value < 0.05
                summary_results.append((model_type, significant, stat, p_value))
            
    
        # Print summary of results
        print(f"{self.dataset} - {self.target} ({metric_name})\n{'='*50}")
        model_order = ['rf', 'svm', 'mlp', 'popphycnn', 'taxonn']
        summary_results = sorted(summary_results, key=lambda x: model_order.index(x[0]))
        for result in summary_results:
            model_type, significant_difference, stat, p_value = result
            # Print the log10 transformed p-value
            print(f"{model_type}: {'Significant' if significant_difference else 'Not Significant'} (p-value: {p_value:.2e})")

    def _visualize_time_elapsed(self):
        dataset_sizes = {
            'rumc_pd': 114,
            'alzbiom_ad': 175,
            'ibd200_type': 174,
            'asd_stage': 60,
            'tbc_pd': 113,
            'hmp2_type': 1158,
            'gd_cohort': 162
        }

        self.results['dataset_size'] = self.results['dataset_target'].apply(lambda x: dataset_sizes[x])
        # self.results = self.results[self.results['Model Type'] != 'treenn']
        self.results = self.results[(self.results['Model Type'] == 'treenn') | (self.results['Model Type'] == 'miostone')]
        self.results = self.results[(self.results['Model Type'] != 'miostone') | 
                                    ((self.results['Model Type'] == 'miostone') & 
                                     self.results['Model Hparams'].apply(lambda params: 
                                                ('node_gate_type', 'concrete') in params and
                                                ('node_dim_func', 'linear') in params and
                                                ('prune_mode', 'taxonomy') in params)) &
                                    self.results['Train Hparams'].apply(lambda params:
                                                                        all('percent_features' not in param for param in params))]
        

        # Set the style for the plots
        sns.set_theme(style="white")

        # Create a plot
        plt.figure(figsize=(20, 10))

        # Use seaborn to create a bar plot
        sns.barplot(x='dataset_target',
                    order=['rumc_pd', 'alzbiom_ad', 'ibd200_type', 'asd_stage', 'tbc_pd', 'gd_cohort', 'hmp2_type'],
                    y='Time Elapsed', 
                    data=self.results,
                    hue='Model Type',
                    # hue_order=['rf', 'svm', 'mlp', 'popphycnn', 'taxonn', 'miostone'],
                    hue_order=['treenn', 'miostone'],
                    # palette=self.model_type_palette,
                    palette=['#1f77b4', '#ffd700'],
                    errorbar="ci")
        
        # Set log scale for the y-axis
        plt.yscale('log')
    
        # Set plot title and labels
        plt.title(f'Computational Cost')
        plt.xlabel('')
        plt.ylabel('Average Training Time (Seconds per Fold)')

        # Set x-axis tick labels
        plt.xticks([0, 1, 2, 3, 4, 5, 6], ['RUMC - PD', 'AlzBiom - AD', 'IBD200 - Type', 'ASD - Stage', 'TBC - PD', 'GD - Cohort', 'HMP2 - Type'])

        # Annotate the bars with the values
        self._annotate_bar_plot(plt.gca())

        # Rename the legend labels and move the legend outside the plot
        handles, labels = plt.gca().get_legend_handles_labels()
        # plt.gca().legend(handles, [self.model_type_labels[label] for label in labels], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().legend(handles, ['MIOSTONE w/o acceleration', 'MIOSTONE w/ acceleration'])
        # plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Adjust layout for better visualization
        plt.tight_layout()

        # Show the plot
        plt.show()

        # Use seaborn to create a scatter plot
        plt.figure(figsize=(12, 10))
        sns.lineplot(data=self.results,
                    x='dataset_size',
                    y='Time Elapsed',
                    hue='Model Type',
                    hue_order=['rf', 'svm', 'mlp', 'popphycnn', 'taxonn', 'miostone'],
                    palette=self.model_type_palette,
                    style='Model Type',
                    markers=True,
                    dashes=False,
                    errorbar='ci')
        
        # Set log scale for both axes
        plt.xscale('log')
        plt.yscale('log')

        # Set plot title and labels
        plt.title(f'Computational Cost')
        plt.xlabel('Dataset Size')
        plt.ylabel('Average Training Time (Seconds per Fold)')

        # Enhance legend
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, [self.model_type_labels[label] for label in labels], loc='center left', bbox_to_anchor=(1, 0.5))

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
    parser.add_argument('--visualize', type=str, required=True, choices=['scores', 'time', 'curves', 'percent_features', 'scores_dot_plot'], help='The type of visualization to generate')
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
    elif args.visualize == 'scores_dot_plot':
        analyzer._visualize_test_scores_dot_plot()