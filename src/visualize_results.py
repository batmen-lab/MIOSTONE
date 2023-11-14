import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import json
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision

from utils import parse_file_name, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Plot the results of the models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, help='Dataset to plot')
    parser.add_argument('--target', type=str, help='Target to plot')
    parser.add_argument('--combine_folds', action='store_true', help='Combine the results of the folds')
    parser.add_argument('--seeds_agg', type=str, default='mean', choices=['mean', 'median', 'none'], help='How to aggregate the results of the seeds')
    parser.add_argument('--metric', type=str, default='accuracy', choices=['accuracy', 'auroc', 'ap'], help='Metric to plot')
    parser.add_argument('--metric_avg', type=str, default='macro', choices=['weighted', 'macro', 'micro'], help='The type of averaging to use for the metric')
    parser.add_argument('--metric_axis', type=str, default='x', choices=['x', 'y'], help='The axis to plot the metric on')
    parser.add_argument('--plot_type', type=str, default='bar', choices=['bar', 'box'], help='The type of plot to generate')
    parser.add_argument('--other_axis', type=str, default='model_name', choices=['model_name', 'dataset:target', 'model', 'transfer_learning_ft_epochs'], help='The axis to plot the metric on')
    parser.add_argument('--hyperparam', type=str, nargs='*', help='List of hyperparameters to marginalize over')
    parser.add_argument('--model_selection', type=str, default='none', choices=['best', 'top_k', 'ablation', 'none'], help='How to select the models to plot')
    parser.add_argument('--top_k', type=int, default=5, help='Top k models to plot (only used if model_selection=top_k)')
    parser.add_argument('--ablation', type=str, choices=['node_gate_type_param', 'node_dim_func', 'taxonomy'], help='Ablation study to plot (only used if model_selection=ablation)')

    return parser.parse_args()

def set_model_selection_param(args):
    if args.model_selection == 'top_k':
        model_selection_param = args.top_k
        hue = 'model'
    elif args.model_selection == 'ablation':
        model_selection_param = args.ablation
        hue = args.ablation
    else:
        model_selection_param = None
        hue = 'model'
    return model_selection_param, hue


def generate_model_name(hyperparameters, chosen_params):
    model_name_parts = []
    
    # Param mapping dictionary
    param_mappings = {
        "rebalance": {"reweight": "rw", "upsample": "up"},
        "mixup_num_samples": lambda x: f"mx{x}",
        "taxonomy": {True: "taxo", False: "phylo"},
        "prune_by_dataset": {True: "pruned", False: "full"},
        "prob_label": {True: "prob", False: "idx"},
        "output_truncation": {True: "ot", False: "oc"},
    }
    
    for param in chosen_params:
        value = hyperparameters.get(param, None)
        if value is not None:
            if param in param_mappings:
                mapping = param_mappings[param]
                if callable(mapping):
                    model_name_parts.append(mapping(value))
                else:
                    model_name_parts.append(mapping.get(value, value))
            else:
                model_name_parts.append(str(value))
    
    return "_".join(model_name_parts)

def get_top_models(df, model, hyperparams, k):
    data = df[df['model'] == model]
    if hyperparams:
        for key, value in hyperparams.items():
            data = data[data[key] == value]
    top_models = data.groupby('model_name')['metric_value'].median().sort_values(ascending=False).index[:k]

    return data[data['model_name'].isin(top_models)]

def compute_metrics(targets, preds, metric_type, metric_avg):
    metrics_functions = {
        'accuracy': Accuracy,
        'auroc': AUROC,
        'ap':AveragePrecision,
    }
    metric = metrics_functions[metric_type](task='multiclass', num_classes=preds.shape[1], average=metric_avg)
    return metric(preds, targets).item()

def metric2label(metric):
    metric_labels = {
        'accuracy': 'Accuracy',
        'auroc': 'Area Under the ROC Curve',
        'ap': 'Area Under the Precision-Recall Curve',
    }
    return metric_labels[metric]

def model2label(model):
    model_labels = {
        'rf': 'Random Forest',
        'mlp': 'Multilayer Perceptron',
        'tree': 'MIOSTONE',
    }
    return model_labels[model]

def node_gate_type2label(node_gate_type):
    node_gate_type_labels = {
        'deterministic_0.0': 'Deterministic Gate (0.0)',
        'deterministic_1.0': 'Deterministic Gate (1.0)',
        'concrete_0.1': 'Binary Concrete Stochastic Gate',
    }
    return node_gate_type_labels[node_gate_type]

def node_dim_func2label(node_dim_func):
    node_dim_func_labels = {
        'const': 'Constant',
        'linear': 'Linear',
    }
    return node_dim_func_labels[node_dim_func]

def taxonomy2label(taxonomy):
    taxonomy_labels = {
        'False': 'Phylogenetic Tree',
        'True': 'Taxonomic Tree',
    }
    return taxonomy_labels[taxonomy]

def transfer_learning2label(transfer_learning):
    transfer_learning_labels = {
        'ft_0': 'Zero-shot',
        'ft_30': 'Fine-tuning',
        'none_0': 'Training from scratch',
    }
    return transfer_learning_labels[transfer_learning]

def preprocess_results(dataset, target, metric, metric_avg, combine_folds, seeds_agg, hyperparam, model_selection, model_selection_param):
    results_dir = f"../results/{dataset}/{target}"
    assert os.path.exists(results_dir), f"Directory {results_dir} does not exist"
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    results_df = []
    column_names = set()
    for json_file in json_files:
            seed, hyperparameters = parse_file_name(json_file)
            with open(os.path.join(results_dir, json_file)) as f:
                data = json.load(f)
                if combine_folds:
                    data = {"combined": {"targets": np.concatenate([fold_data["targets"] for fold_data in data.values()]), 
                                        "predictions": np.concatenate([fold_data["predictions"] for fold_data in data.values()])}}
                for fold, fold_data in data.items():
                    targets = torch.tensor(fold_data['targets'])
                    preds = torch.tensor(fold_data['predictions'])
                    metric_value = compute_metrics(targets, preds, metric, metric_avg)
                    
                    # Choose which hyperparameters you want in the model's name
                    # chosen_params_rf = ["model", "preprocess", "mixup_num_samples", "mixup_alpha", "taxonomy", "prune_by_dataset"]
                    chosen_params_rf = ["model", "prune_by_dataset"]
                    chosen_params_mlp = ["model", "prune_by_dataset", "loss", "mlp_type"]
                    chosen_params_tree = ["model", "taxonomy", "prune_by_dataset", "loss", "node_min_dim", "node_dim_func", "node_dim_func_param", "node_gate_type", "node_gate_param", "output_depth"]
                    # chosen_params_nn = ["model", "batch_size", "loss", "focal_alpha", "focal_gamma", "gate_type", "gate_param", "node_bottleneck_dim"]
                    if hyperparameters["model"] == "rf":
                        model_name = generate_model_name(hyperparameters, chosen_params_rf)
                    elif hyperparameters["model"] == "mlp":
                        model_name = generate_model_name(hyperparameters, chosen_params_mlp)
                    else:
                        model_name = generate_model_name(hyperparameters, chosen_params_tree)
                    new_row = {"dataset:target": dataset + ":" + target, "seed": seed, "fold": fold, "model_name": model_name, "metric_value": metric_value, **hyperparameters}
                    results_df.append(new_row)
                    column_names.update(new_row.keys())

    # Ensuring all dictionaries have the same set of keys
    for i, row in enumerate(results_df):
        for key in column_names:
            if key not in row:
                results_df[i][key] = pd.NA

    # Converting to a dataframe
    results_df = pd.DataFrame(results_df)

    # Aggregating the results of the seeds if needed
    if seeds_agg != 'none':
        results_df = results_df.groupby('model_name').agg({
            'metric_value': seeds_agg,
            **{col: 'first' for col in column_names if col not in ['metric_value', 'model_name']}
        }).reset_index()

    if hyperparam: # Adding a column for the hyperparameter combination
        results_df["_".join(hyperparam)] = results_df.apply(lambda row: "_".join(row[hyperparam].astype(str)), axis=1)
        if '_'.join(hyperparam) == 'transfer_learning_ft_epochs':
            results_df['_'.join(hyperparam)] = results_df.apply(lambda row: transfer_learning2label(row['_'.join(hyperparam)]), axis=1)
    
    if model_selection != 'none':  # Selecting models based on the model selection criteria if needed
        if model_selection == 'top_k':
            model_configs = {
                'rf': ('rf', {}),
                'mlp': ('mlp', {}),
                'tree': ('tree', {}),
            }
            k = model_selection_param
        elif model_selection == 'best':
            model_configs = {
                'rf': ('rf', {}),
                'mlp': ('mlp', {}),
                'tree_concrete': ('tree', {'node_gate_type': 'concrete', 'node_gate_param': 0.1, 'node_dim_func': 'linear'}),
            }
            k = 1
        elif model_selection == 'ablation':
            if model_selection_param == 'node_gate_type_param':
                results_df['node_gate_type_param'] = results_df['node_gate_type'] + '_' + results_df['node_gate_param'].astype(str)
                model_configs = {
                    'tree_deterministic_0': ('tree', {'node_gate_type_param': 'deterministic_0.0', 'node_dim_func': 'linear'}),
                    'tree_deterministic_1': ('tree', {'node_gate_type_param': 'deterministic_1.0', 'node_dim_func': 'linear'}),
                    'tree_concrete': ('tree', {'node_gate_type_param': 'concrete_0.1', 'node_dim_func': 'linear'}),
                }
            elif model_selection_param == 'node_dim_func':
                model_configs = {
                    'tree_const': ('tree', {'node_gate_type': 'concrete', 'node_gate_param': 0.1, 'node_dim_func': 'const'}),
                    'tree_linear': ('tree', {'node_gate_type': 'concrete', 'node_gate_param': 0.1, 'node_dim_func': 'linear'}),
                }
            elif model_selection_param == 'taxonomy':
                model_configs = {
                    'tree_phylo': ('tree', {'node_gate_type': 'concrete', 'node_gate_param': 0.1, 'node_dim_func': 'linear', 'taxonomy': False}),
                    'tree_taxo': ('tree', {'node_gate_type': 'concrete', 'node_gate_param': 0.1, 'node_dim_func': 'linear', 'taxonomy': True}),
                }
            else:
                raise ValueError(f"Unknown ablation study {model_selection_param}")
            k = 1

        results_df = pd.concat([
            get_top_models(results_df, model, hyperparams, k)
            for _, (model, hyperparams) in model_configs.items()
        ])

    return results_df

def plot_model_metrics(results_df, metric, seeds_agg, hue, plot_type, metric_axis, other_axis):
    # Set the figure size
    plt.figure(figsize=(15, 15))

    # Determine axes and order
    if metric_axis == 'x':
        x = 'metric_value'
        y = other_axis
    else:
        x = other_axis
        y = 'metric_value'

    # Set the color palette
    palette, hue_order = set_palette_and_order(results_df, hue)

    # Generate the plot
    ax = generate_plot(results_df, hue, palette, hue_order, plot_type, x, y)

    # Set axis limits
    set_axis_limits(ax, results_df['metric_value'], metric_axis)

    # Set axis labels
    set_axis_labels(ax, metric, metric_axis)

    # Bar plot specific settings
    if plot_type == 'bar':
        # Annotate bars with metric values
        annotate_bars(ax, seeds_agg, metric_axis)

        # Set the legend
        set_legend(ax, hue, metric_axis)

    plt.tight_layout()
    plt.show()

def generate_plot(results_df, hue, palette, hue_order, plot_type, x, y):
    if plot_type == 'bar':
        if ('model' in [x, y] and hue == 'model'):
            dodge = False
            order = ['rf', 'mlp', 'tree']
        else:
            dodge = True
            order = None
        ax = sns.barplot(x=x, y=y, hue=hue, palette=palette, hue_order=hue_order, errorbar="sd", errwidth=2, data=results_df, dodge=dodge, order=order)
    else:
        marker_shapes = ['o', 's', '^']

        # Create a unique shape for each fold
        shape_dict = {fold: shape for fold, shape in zip(results_df['fold'].unique(), marker_shapes)}

        # Create the boxplot
        ax = sns.boxplot(x=x, y=y, hue=hue, palette=palette, data=results_df, dodge=False)

        # Add stripplot for each fold
        seed_legend = True
        for fold, shape in shape_dict.items():
            sns.stripplot(x=x, y=y, hue='seed', data=results_df[results_df['fold'] == fold], 
                        marker=shape, size=10, jitter=False, dodge=False, ax=ax, legend=seed_legend)
            seed_legend = False

        # Create custom legend for the scatterplot's shape
        custom_shape_legend = [mlines.Line2D([], [], color='black', marker=shape, 
                                             linestyle='None', markersize=10, label=f"{fold}") 
                               for fold, shape in shape_dict.items()]
        

        leg = ax.legend(title='Seed', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Legend for the scatterplot's shape
        ax.add_artist(leg) 
        plt.legend(handles=custom_shape_legend, title='fold', bbox_to_anchor=(1.05, 0.5), loc='upper left')

    return ax

def set_palette_and_order(results_df, hue):
    if hue == 'model':
        palette = {'rf': 'tab:green', 'mlp': 'tab:orange', 'tree': 'tab:blue'}
        order = ['rf', 'mlp', 'tree']
    elif hue == 'node_gate_type_param':
        palette = {'deterministic_0.0': 'tab:green', 'deterministic_1.0': 'tab:orange', 'concrete_0.1': 'tab:blue'}
        order = ['deterministic_0.0', 'deterministic_1.0', 'concrete_0.1']
    elif hue == 'node_dim_func':
        palette = {'const': 'tab:orange', 'linear': 'tab:blue'}
        order = ['const', 'linear']
    elif hue == 'taxonomy':
        palette = {False: 'tab:orange', True: 'tab:blue'}
        order = [False, True]
    else:
        palette = sns.color_palette("tab10", n_colors=results_df[hue].nunique())
        order = None
    return palette, order

def set_axis_limits(ax, metric_values, metric_axis):
    # min_val = metric_values.min()
    min_val = 0.5
    max_val = 1.0
    # max_val = metric_values.max()
    # margin = (max_val - min_val) * 0.15
    margin = 0.02
    if metric_axis == 'x':
        ax.set_xlim(min_val - margin, max_val + margin)
    else:
        ax.set_ylim(min_val - margin, max_val + margin)

def set_axis_labels(ax, metric, metric_axis):
    if metric_axis == 'x':
        x_label = metric2label(metric)
        y_label = ''
    else:
        y_label = metric2label(metric)
        x_label = ''
    ax.set_xlabel(x_label, fontsize=22)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.set_ylabel(y_label, fontsize=22)
    ax.yaxis.set_tick_params(labelsize=16)

def annotate_bars(ax, seeds_agg, metric_axis):
    if metric_axis == 'x':
        offset = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    else:
        offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    if seeds_agg == 'none':
        if metric_axis == 'x':
            for p, err_bar in zip(ax.patches, ax.lines):
                bar_length = p.get_width()
                err_bar_length = err_bar.get_xdata()[1] - err_bar.get_xdata()[0]
                text_position = bar_length + err_bar_length / 2
                position = (text_position + offset, p.get_y() + p.get_height() / 2.)
                ax.annotate(f"{bar_length:.2f}", xy=position, ha='center', va='center', fontsize=16, color='black')
        else:
            for p, err_bar in zip(ax.patches, ax.lines):
                bar_length = p.get_height()
                err_bar_length = err_bar.get_ydata()[1] - err_bar.get_ydata()[0]
                text_position = bar_length + err_bar_length / 2
                position = (p.get_x() + p.get_width() / 2., text_position + offset)
                ax.annotate(f"{bar_length:.2f}", xy=position, ha='center', va='center', fontsize=16, color='black')
    else:
        if metric_axis == 'x':
            for p in ax.patches:
                bar_length = p.get_width()
                position = (bar_length + offset, p.get_y() + p.get_height() / 2.)
                ax.annotate(f"{bar_length:.2f}", xy=position, ha='center', va='center', fontsize=16, color='black')
        else:
            for p in ax.patches:
                bar_length = p.get_height()
                position = (p.get_x() + p.get_width() / 2., bar_length + offset)
                ax.annotate(f"{bar_length:.2f}", xy=position, ha='center', va='center', fontsize=16, color='black')

def set_legend(ax, hue, metric_axis):
    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()
        if metric_axis == 'x':
            loc = 'lower right'
        else:
            loc = 'upper left'
        if hue == 'model':
            labels = [model2label(label) for label in labels]
        elif hue == 'node_gate_type_param':
            labels = [node_gate_type2label(label) for label in labels]
        elif hue == 'node_dim_func':
            labels = [node_dim_func2label(label) for label in labels]
        elif hue == 'taxonomy':
            labels = [taxonomy2label(label) for label in labels]
        ax.legend(handles[:len(set(labels))], labels[:len(set(labels))], loc=loc, fontsize=16)


if __name__ == "__main__":
    # Argument parser setup
    args = parse_args()

    # Set the random seed
    set_seed(args.seed)
    
    # Set the model selection parameters and hue
    model_selection_param, hue = set_model_selection_param(args)

    # Preprocess the results and plot them
    if args.dataset and args.target:
        results_df = preprocess_results(args.dataset, args.target, args.metric, args.metric_avg, args.combine_folds, 
                                        args.seeds_agg, args.hyperparam, args.model_selection, model_selection_param)
        plot_model_metrics(results_df, args.metric, args.seeds_agg, hue, args.plot_type, args.metric_axis, args.other_axis)
    else:
        dataset_targets = [(dataset, target) for dataset in os.listdir('../results') for target in os.listdir(f'../results/{dataset}') if not target.endswith('tl')]
        assert dataset_targets, "No datasets found"
        results_dfs = [preprocess_results(dataset, target, args.metric, args.metric_avg, args.combine_folds, 
                                          args.seeds_agg, args.hyperparam, args.model_selection, model_selection_param) 
                                          for dataset, target in dataset_targets]
        results_df = pd.concat(results_dfs, ignore_index=True)
        plot_model_metrics(results_df, args.metric, args.seeds_agg, hue, args.plot_type, args.metric_axis, other_axis='dataset:target')
