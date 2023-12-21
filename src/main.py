import argparse
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics
from tqdm import tqdm

from train import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Run the MIOSTONE Training Pipeline")

    # Base parameters
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    # Data parameters
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--target', type=str, required=True, help='Target to predict')

    # Model parameters
    parser.add_argument('--model_type', type=str, required=True, choices=['mlp', 'tree'], help='Type of model to train')

    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--k_folds', type=int, default=3, help='Number of folds for cross-validation')

    # Mixup parameters
    parser.add_argument('--mixup_num_samples', type=int, help='Number of samples for mixup')
    parser.add_argument('--mixup_q_interval_start', type=float, help='Start of the quantile interval for mixup')
    parser.add_argument('--mixup_q_interval_end', type=float, help='End of the quantile interval for mixup')

    args = parser.parse_args()

    # Model parameters
    if args.model_type == 'mlp':
        model_hparams = {}
    elif args.model_type == 'tree':
        model_hparams = {
            'node_min_dim': 1,
            'node_dim_func': 'linear',
            'node_dim_func_param': 0.95,
            'node_gate_type': 'concrete',
            'node_gate_param': 0.05,
        }

    # Training parameters
    train_hparams = {
        'max_epochs': args.max_epochs,
        'k_folds': args.k_folds,
        'batch_size': 512
    }

    # Mixup parameters
    mixup_hparams = {}
    if args.mixup_num_samples is not None and args.mixup_q_interval_start is not None and args.mixup_q_interval_end is not None:
        mixup_hparams = {
            'num_samples': args.mixup_num_samples,
            'q_interval': (args.mixup_q_interval_start, args.mixup_q_interval_end)
        }

    # Initialize the training pipeline
    pipeline = TrainingPipeline(seed=args.seed)

    # Load the data and tree
    pipeline.load_data_and_tree(dataset=args.dataset, target=args.target, preprocess=False)

    # Run the training pipeline for the baseline
    pipeline.train(model_type=args.model_type,
                   model_hparams=model_hparams,
                   train_hparams=train_hparams,
                   mixup_hparams=mixup_hparams)

    '''
    # Run the training pipeline f
    num_samples_list = [100]
    q_intervals = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for num_samples, q_interval in tqdm(product(num_samples_list, q_intervals), total=len(num_samples_list) * len(q_intervals)):
        mixup_hparams = {
            'num_samples': num_samples,
            'q_interval': q_interval
        }
        pipeline.train(model_type=args.model_type, 
                       model_hparams=model_hparams, 
                       train_hparams=train_hparams,
                       mixup_hparams=mixup_hparams)
    
    # Create DataFrame from results
    df_results = pd.DataFrame(pipeline.results)

    # Initialize metrics
    num_classes = len(np.unique(pipeline.dataset.y))
    auroc_metric = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro')
    auprc_metric = torchmetrics.AveragePrecision(task='multiclass', num_classes=num_classes, average='macro')

    # Function to calculate metrics
    def calculate_metrics(metric, true_labels, logits):
        true_labels = torch.tensor(true_labels)
        logits = torch.tensor(logits)
        return metric(logits, true_labels).item()

    # Apply metrics calculation
    df_results['AUROC'] = df_results.apply(lambda x: calculate_metrics(auroc_metric, x['True Labels'], x['Logits']), axis=1)
    df_results['AUPRC'] = df_results.apply(lambda x: calculate_metrics(auprc_metric, x['True Labels'], x['Logits']), axis=1)

    # Aggregate results
    avg_metrics = df_results.groupby(['Num Samples', 'Quantile']).mean()[['AUROC', 'AUPRC']].reset_index()

    # Melt DataFrame for plotting
    plot_data = pd.melt(avg_metrics, id_vars=['Num Samples', 'Quantile'], value_vars=['AUROC', 'AUPRC'], var_name='Metric', value_name='Value')

    # Plotting
    plt.figure(figsize=(12, 6))

    # Order of quantile intervals
    quantile_order = ['(0.0, 0.2)', '(0.2, 0.4)', '(0.4, 0.6)', '(0.6, 0.8)', '(0.8, 1.0)', 'Baseline']

    # Subplot for AUROC
    plt.subplot(1, 2, 1)
    sns.pointplot(data=plot_data[plot_data['Metric'] == 'AUROC'], x='Quantile', y='Value', hue='Num Samples', order=quantile_order)
    plt.title('AUROC across Quantile Intervals')
    plt.xlabel('Quantile Interval')
    plt.ylabel('AUROC')

    # Subplot for AUPRC
    plt.subplot(1, 2, 2)
    sns.pointplot(data=plot_data[plot_data['Metric'] == 'AUPRC'], x='Quantile', y='Value', hue='Num Samples', order=quantile_order)
    plt.title('AUPRC across Quantile Intervals')
    plt.xlabel('Quantile Interval')
    plt.ylabel('AUPRC')

    plt.tight_layout()
    plt.show()
    '''
        
if __name__ == "__main__":
    main()