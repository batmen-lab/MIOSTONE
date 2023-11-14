import argparse
import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim as optim
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision
from tqdm import tqdm

from model import TreeNN
from utils import (load_data, parse_file_name, preprocess_data, set_device,
                   set_seed)


def parse_args():
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='PhyloNets')

    # Dataset Parameters
    data_group = parser.add_argument_group('Dataset Parameters')
    data_group.add_argument('--dataset', default='IBD200', type=str, help='Dataset to use')
    data_group.add_argument('--target', default='Type', type=str, help='Target label for predictions')
    data_group.add_argument('--prob_label', action='store_true', help='Use probabilistic labels')
    data_group.add_argument('--rebalance', default='reweight', choices=['none', 'reweight', 'upsample'], type=str, help='Method for handling class imbalance')
    data_group.add_argument('--preprocess', default='clr', choices=['clr', 'log'], type=str, help='Preprocessing method')
    data_group.add_argument('--phylogeny', type=str, default="WoL2", choices=["WoL", "WoL2"], help="Phylogenetic tree to use.")
    data_group.add_argument('--taxonomy', action='store_true', help='Use the taxonomic tree instead of the phylogenetic tree')
    data_group.add_argument('--prune_by_dataset', action='store_true', help='Prune the tree based on the dataset')
    data_group.add_argument('--tree_dataset', default='none', type=str, help='Dataset to use for the tree. If not specified, the same dataset as the main dataset is used.')
    data_group.add_argument('--tree_target', default='none', type=str, help='Target label for the tree. If not specified, the same target as the main dataset is used.')
    data_group.add_argument('--prune_by_num_leaves', default=-1, type=int, help='Number of leaves to keep after pruning. If negative, no pruning is performed.')
    data_group.add_argument('--collapse_threshold', default=-1.0, type=float, help='Threshold for collapsing nodes. If negative, no collapsing is performed.')

    # Model Parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--model', default="tree", choices=["rf", "mlp", "tree"], type=str, help='Baseline model')
    model_group.add_argument('--mlp_type', default="pyramid", choices=["uniform", "pyramid"], type=str, help='MLP type')
    model_group.add_argument('--node_min_dim', default=1, type=int, help='Minimum dimension of the node embedding')
    model_group.add_argument('--node_dim_func', default="linear", choices=["const", "linear"], type=str, help='Function for computing the node embedding dimension')
    model_group.add_argument('--node_dim_func_param', default=0.95, type=float, help='Parameter for the node embedding dimension function')
    model_group.add_argument('--node_gate_type', default='concrete', choices=['deterministic', 'gaussian', 'concrete'], type=str, help='Gate type for the node embedding')
    model_group.add_argument('--node_gate_param', default=0.1, type=float, help='Gate parameter for the node embedding')
    model_group.add_argument('--output_depth', default=0, type=int, help='Depth of the output nodes')
    model_group.add_argument('--output_truncation', action='store_true', help='Truncate the nodes with depth greater than the output depth or concatenate them to the output nodes')
    
    # Transfer Learning Parameters
    transfer_group = parser.add_argument_group('Transfer Learning Parameters')
    transfer_group.add_argument('--transfer_learning', default="none", choices=["zs", "ft", "tfs", "none"], type=str, help='Transfer learning method')
    transfer_group.add_argument('--pretrained_model_path', default="", type=str, help='Path to the pre-trained model')
    transfer_group.add_argument('--ft_epochs', default=30, type=int, help='Total fine-tuning epochs')

    # Training Parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--k_fold', default=3, type=int, help='Number of folds for cross-validation')
    training_group.add_argument('--batch_size', default=386, type=int, help='Training batch size')
    training_group.add_argument('--epochs', default=50, type=int, help='Total training epochs')
    training_group.add_argument('--loss', default='ce', choices=['ce', 'focal'], type=str, help='Loss function')
    training_group.add_argument('--focal_gamma', default=2.0, type=float, help='Gamma parameter for focal loss')
    training_group.add_argument('--mixup_num_samples', default=0, type=int, help='Number of mixup samples')
    training_group.add_argument('--mixup_alpha', default=0.0, type=float, help='Alpha parameter for mixup')
    training_group.add_argument('--mixup_remix', action='store_true', help='Use Remix instead of original Mixup')
    training_group.add_argument('--mixup_remix_tau', default=0.5, type=float, help='Tau parameter for Remix')
    training_group.add_argument('--mixup_remix_kappa', default=1.5, type=int, help='Kappa parameter for Remix')
    training_group.add_argument('--metrics', default=['accuracy', 'auroc', 'ap'], nargs='+', type=str, help='Metrics to compute')

    # General Parameters
    parser.add_argument('--seed', default=3, type=int, help='Random seed for reproducibility')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'], type=str, help='Computing device to use: cpu, cuda, or mps')

    # Visualization
    visual_group = parser.add_argument_group('Visualization')
    visual_group.add_argument('--plot_loss', action='store_true', help='Generate a loss curve')

    # Saving Parameters
    saving_group = parser.add_argument_group('Saving Parameters')
    saving_group.add_argument('--save_model', action='store_true', help='Save the model')

    return parser.parse_args()

def set_args_for_transfer_learning(args, pretrained_seed, hyperparameters):
    args.pretrained_seed = pretrained_seed
    for key, value in hyperparameters.items():
        if key not in ["transfer_learning", "ft_epochs", "pretrained_seed"]:
            setattr(args, key, value)


def set_results_dir(dataset, target):
    results_dir = f"../results/{dataset}/{target}"
    losses_dir = f"{results_dir}/losses"
    models_dir = f"{results_dir}/models"
    for dir in [results_dir, losses_dir, models_dir]: 
        if not os.path.exists(dir):
            os.makedirs(dir)
    return results_dir, losses_dir, models_dir

def set_file_name(args):
    file_name = f"{args.seed}_{args.model}_{args.preprocess}_{args.rebalance}"
    if args.model == "rf":
        if args.mixup_num_samples > 0:
            file_name += f"_mx{args.mixup_num_samples}_{args.mixup_alpha}"
        if args.taxonomy:
            file_name += "_tx"
        if args.tree_dataset != "none" and args.tree_target != "none":
            file_name += f"_td{args.tree_dataset}_{args.tree_target}"
        if args.prune_by_dataset:
            file_name += "_pd"
        if args.prob_label:
            file_name += "_pl"
        if args.save_model and args.transfer_learning == "none":
            file_name += "_saved"
        if args.transfer_learning == "zs":
            file_name += f"_zs{args.pretrained_seed}"
    elif args.model == "mlp":
        file_name += f"_{args.batch_size}_{args.epochs}_{args.loss}_{args.mlp_type}"
        if args.mixup_num_samples > 0:
            file_name += f"_mx{args.mixup_num_samples}_{args.mixup_alpha}"
            if args.mixup_remix:
                file_name += f"_rmx{args.mixup_remix_tau}_{args.mixup_remix_kappa}"
        if args.taxonomy:
            file_name += "_tx"
        if args.tree_dataset != "none" and args.tree_target != "none":
            file_name += f"_td{args.tree_dataset}_{args.tree_target}"
        if args.prune_by_dataset:
            file_name += "_pd"
        if args.prob_label:
            file_name += "_pl"
        if args.loss == "focal":
            file_name += f"_fg{args.focal_gamma}"
        if args.save_model and args.transfer_learning == "none":
            file_name += "_saved"
        if args.transfer_learning == "zs":
            file_name += f"_zs{args.pretrained_seed}"
        if args.transfer_learning == "ft":
            file_name += f"_ft{args.pretrained_seed}_{args.ft_epochs}"
    elif args.model == "tree":
        file_name += f"_{args.batch_size}_{args.epochs}_{args.loss}_{args.node_min_dim}_{args.node_dim_func}_{args.node_dim_func_param}_{args.node_gate_type}_{args.node_gate_param}_{args.output_depth}"
        if args.mixup_num_samples > 0:
            file_name += f"_mx{args.mixup_num_samples}_{args.mixup_alpha}"
            if args.mixup_remix:
                file_name += f"_rmx{args.mixup_remix_tau}_{args.mixup_remix_kappa}"
        if args.taxonomy:
            file_name += "_tx"
        if args.tree_dataset != "none" and args.tree_target != "none":
            file_name += f"_td{args.tree_dataset}_{args.tree_target}"
        if args.prune_by_dataset:
            file_name += "_pd"
        if args.prob_label:
            file_name += "_pl"
        if args.output_truncation:
            file_name += "_ot"
        if args.loss == "focal":
            file_name += f"_fg{args.focal_gamma}"
        if args.save_model and args.transfer_learning == "none":
            file_name += "_saved"
        if args.transfer_learning == "zs":
            file_name += f"_zs{args.pretrained_seed}"
        if args.transfer_learning == "ft":
            file_name += f"_ft{args.pretrained_seed}_{args.ft_epochs}"

    print(f"File name: {file_name}")
    return file_name

def load_and_preprocess_data(args):
    # Load the data
    data, metadata = load_data(f"../data/{args.dataset}", args.dataset)
    
    # Preprocess the data
    custom_tree, dataset, pretrained_seed, hyperparameters = None, None, None, None
    if args.transfer_learning != "none":
        if args.tree_dataset == "none" or args.tree_target == "none":
            raise ValueError("Please specify the dataset and target used for pre-training. Use the --tree_dataset and --tree_target arguments.")
        pretrained_seed, hyperparameters = parse_file_name(os.path.basename(args.pretrained_model_path))
        pretrained_data, pretrained_metadata = load_data(f"../data/{args.tree_dataset}", args.tree_dataset)
        prune_by_dataset = True if hyperparameters["tree_dataset"] and hyperparameters["tree_target"] else hyperparameters["prune_by_dataset"]
        custom_tree, _ = preprocess_data(pretrained_data, pretrained_metadata, args.tree_dataset, args.tree_target, args.phylogeny, 
                                                          hyperparameters["taxonomy"], hyperparameters["prob_label"], prune_by_dataset=prune_by_dataset)
    else:
        if args.tree_dataset != "none" and args.tree_target != "none":
            tree_data, tree_metadata = load_data(f"../data/{args.tree_dataset}", args.tree_dataset)
            custom_tree, _ = preprocess_data(tree_data, tree_metadata, args.tree_dataset, args.tree_target, 
                                             args.phylogeny, args.taxonomy, args.prob_label, prune_by_dataset=True)
            
    custom_tree, dataset = preprocess_data(data, metadata, args.dataset, args.target, args.phylogeny, args.taxonomy, 
                                            args.prob_label, args.prune_by_dataset, args.prune_by_num_leaves, custom_tree)
    
    return custom_tree, dataset, pretrained_seed, hyperparameters

def run_epoch(model, device, dataloader, criterion, l0_weight=0.0, optimizer=None, scheduler=None):
    running_loss = 0.0
    running_l0_reg = 0.0

    all_targets = []
    all_preds = []

    # Check if the model is in training mode
    is_training = model.training

    # Use torch.no_grad if model is in evaluation mode
    with torch.set_grad_enabled(is_training):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs
            l0_reg = model.get_total_l0_reg() if type(model) == TreeNN else torch.tensor(0.0)
            loss = criterion(preds, targets)

            if is_training:
                total_loss = loss + l0_weight * l0_reg
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_l0_reg += l0_reg.item() * inputs.size(0)

            # Convert one-hot targets to class labels if necessary
            if targets.dim() > 1:
                targets = torch.argmax(targets, dim=1)

            all_targets.append(targets.detach().cpu())
            all_preds.append(preds.detach().cpu())

    if is_training and scheduler is not None:
        scheduler.step()

    loss = running_loss / len(dataloader.sampler)
    l0_reg = running_l0_reg / len(dataloader.sampler)
    all_targets = torch.cat(all_targets, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    return loss, l0_reg, all_targets, all_preds

def train(model, device, dataloader, criterion, l0_weight, optimizer, scheduler):
    model.train()
    return run_epoch(model, device, dataloader, criterion, l0_weight, optimizer, scheduler)

def evaluate(model, device, dataloader, criterion):
    model.eval()
    return run_epoch(model, device, dataloader, criterion)

def train_model(args, device, dataset, custom_tree, file_name, results_dir, losses_dir, models_dir):
     # Create k-fold object
    if args.k_fold > 1:
        kfold = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
        split = kfold.split(dataset.X, dataset.non_prob_y)
    else:
        split = [(range(len(dataset.X)), range(len(dataset.X)))]

    # Initialize the results dictionary
    results = {}

    # Instead of directly splitting the dataset into training and validation,
    # we're going to define 'k' splits.
    for fold, (train_ids, val_ids) in enumerate(split):
        print(f"FOLD {fold}")
        print("--------------------------------")
        
        # Set up the results dictionary for this fold 
        results[f'fold_{fold}'] = {"targets": None, "predictions": None}

        # Create a copy of the dataset
        dataset_copy = copy.deepcopy(dataset)

        # Handle class imbalance
        if args.rebalance == 'upsample':
            dataset_copy, train_ids, val_ids = dataset_copy.upsample(train_ids, val_ids)
        elif args.rebalance == 'reweight':
            class_weight = compute_class_weight('balanced', classes=np.unique(dataset_copy.non_prob_y[train_ids]), y=dataset_copy.non_prob_y[train_ids])
            if args.model == "rf":
                class_weight = {i: class_weight[i] for i in range(len(class_weight))}
            else:
                class_weight = torch.tensor(class_weight, dtype=torch.float32).to(device)
        else:
            class_weight = None

        # Add mixed samples to the training set if mixup is enabled
        if args.mixup_num_samples > 0:
            dataset_copy, train_ids, val_ids = dataset_copy.mixup(train_ids, val_ids, args.mixup_num_samples, args.mixup_alpha, 
                                                                  args.mixup_remix, args.mixup_remix_tau, args.mixup_remix_kappa)

        # Preprocess the data
        dataset_copy = dataset_copy.preprocess(args.preprocess)
        # custom_tree.generate_histograms(dataset_copy, min_depth=4, max_depth=custom_tree.max_depth)
        
        # Define the model
        if args.model == "rf":
            # Split the data into training and validation
            train_X = dataset_copy.X.iloc[train_ids]
            train_y = dataset_copy.y.iloc[train_ids]
            val_X = dataset_copy.X.iloc[val_ids]
            val_y = dataset_copy.y.iloc[val_ids]

            # Define the model and fit it to the training data
            if args.transfer_learning in ["zs", "ft"]:
                model = load(args.pretrained_model_path)
            else:
                model = RandomForestClassifier(random_state=args.seed, class_weight=class_weight) if args.rebalance == 'reweight' else RandomForestClassifier(random_state=args.seed)
                model.fit(train_X, train_y) 

            # Evaluate the model       
            val_y = torch.tensor(val_y.values, dtype=torch.long).to(device)
            val_preds = torch.tensor(model.predict_proba(val_X), dtype=torch.float32).to(device)

            # Save the model
            if args.save_model:
                dump(model, f"{models_dir}/{file_name}_fold{fold}.joblib")
        else:
             # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            
            # Create dataloaders for train and validation
            train_dataloader = DataLoader(dataset_copy, batch_size=args.batch_size, sampler=train_subsampler)
            val_dataloader = DataLoader(dataset_copy, batch_size=args.batch_size, sampler=val_subsampler)

            # Define the model
            if args.transfer_learning in ["zs", "ft"]:
                model = torch.load(args.pretrained_model_path)
            else:
                if args.model == "mlp":
                    if args.mlp_type == "pyramid":
                        model = nn.Sequential(
                            nn.Linear(dataset_copy.X.shape[1], dataset_copy.X.shape[1] // 2),
                            nn.ReLU(),
                            nn.Linear(dataset_copy.X.shape[1] // 2, dataset_copy.X.shape[1] // 4),
                            nn.ReLU(),
                            nn.Linear(dataset_copy.X.shape[1] // 4, dataset_copy.num_classes)
                        )
                    else:
                        model = nn.Sequential(
                            nn.Linear(dataset_copy.X.shape[1], dataset_copy.X.shape[1]),
                            nn.ReLU(),
                            nn.Linear(dataset_copy.X.shape[1], dataset_copy.X.shape[1]),
                            nn.ReLU(),
                            nn.Linear(dataset_copy.X.shape[1], dataset_copy.num_classes)
                        )
                    model.to(device)
                else:
                    output_dim = dataset_copy.num_classes
                    model = TreeNN(device, custom_tree, args.node_min_dim, 
                                   args.node_dim_func, args.node_dim_func_param, 
                                   args.node_gate_type, args.node_gate_param,
                                   output_dim, args.output_depth, args.output_truncation)              

            # Define the loss function
            if args.loss == 'ce':
                train_criterion = nn.CrossEntropyLoss(weight=class_weight)
            elif args.loss == 'focal':
                train_criterion = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='FocalLoss', alpha=class_weight, gamma=args.focal_gamma, reduction='mean', force_reload=False, trust_repo=True)
            
            val_criterion = nn.CrossEntropyLoss()
            
            # Define the optimizer
            optimizer = optim.Adam(model.parameters())

            if args.transfer_learning != "zs":
                # Training loop
                train_losses  = []
                val_losses = []
                l0_weight = 1.0
                epochs = args.ft_epochs if args.transfer_learning == "ft" else args.epochs
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                # scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs // 2)
                interval = 10
                for epoch in tqdm(range(epochs)):
                    train_loss, train_l0, train_y, train_preds = train(model, device, train_dataloader, train_criterion, l0_weight, optimizer, scheduler)
                    val_loss, val_l0, val_y, val_preds = evaluate(model, device, val_dataloader, val_criterion)

                    # Save the losses
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    
                    # Report the metrics every 'interval' epochs
                    if epoch % interval == 0:
                        print(f"Epoch {epoch + 1}/{epochs}")
                        print(f"Training loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}")
                        print(f"Training L0 reg: {train_l0:.4f} | Validation L0 reg: {val_l0:.4f}")
                        train_metrics = {metric: compute_metric(train_y, train_preds, metric) for metric in args.metrics}
                        val_metrics = {metric: compute_metric(val_y, val_preds, metric) for metric in args.metrics}
                        for metric in args.metrics:
                            for c in range(dataset_copy.num_classes):
                                print(f"Training {metric} for class {c}: {train_metrics[metric][c]:.4f} | Validation {metric} for class {c}: {val_metrics[metric][c]:.4f}")

                # Plot the loss curve
                if args.plot_loss:
                    plot_loss_curve(train_losses, val_losses, f"{losses_dir}/{file_name}_fold{fold}.png")

                # Save the model
                if args.save_model:
                    torch.save(model, f"{models_dir}/{file_name}_fold{fold}.pt")

            # Evaluate the model
            val_loss, val_l0, val_y, val_preds = evaluate(model, device, val_dataloader, val_criterion)

        # Save the results
        results[f'fold_{fold}']["targets"] = val_y
        results[f'fold_{fold}']["predictions"] = val_preds

        # Compute the metrics for this fold and add them to the results
        print("--------------------------------")
        val_metrics = {metric: compute_metric(val_y, val_preds, metric) for metric in args.metrics}
        for metric in args.metrics:
            for c in range(dataset_copy.num_classes):
                print(f"Validation {metric} for class {c}: {val_metrics[metric][c]:.4f}")

    # Save the results
    with open(f"{results_dir}/{file_name}.json", 'w') as f:
        for fold in results:
            results[fold]["targets"] = results[fold]["targets"].tolist()
            results[fold]["predictions"] = results[fold]["predictions"].tolist()
        json.dump(results, f, indent=4)

def compute_metric(targets, preds, metric_type):
    metrics_functions = {
        'accuracy': Accuracy,
        'auroc': AUROC,
        'ap':AveragePrecision,
    }
    metric = metrics_functions[metric_type](task='multiclass', num_classes=preds.shape[1], average='none')
    return metric(preds, targets)

def plot_loss_curve(train_losses, val_losses, save_path):
    # Set the Seaborn theme
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 6))
    
    # Plot losses
    epochs = len(train_losses)
    sns.lineplot(x=range(epochs), y=train_losses, label='Training Loss', color='blue')
    sns.lineplot(x=range(epochs), y=val_losses, label='Validation Loss', color='red')
    
    # Fill between the loss curves
    plt.fill_between(range(epochs), train_losses, val_losses, color='grey', alpha=0.1)
    
    # Annotate the epoch with the lowest validation loss
    min_val_loss = min(val_losses)
    min_val_loss_epoch = val_losses.index(min_val_loss)
    plt.scatter(min_val_loss_epoch, min_val_loss, color='red', zorder=5)
    plt.text(min_val_loss_epoch, min_val_loss, 'Lowest Val Loss', fontsize=9, ha='right', va='bottom')
    
    # Titles, labels and other plot components
    title = 'Training & Validation Loss Curve '
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Save the plot
    plt.savefig(save_path)


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = set_device(args.device)
    print(f"Using device: {device}")

    # Load and preprocess the data
    custom_tree, dataset, pretrained_seed, hyperparameters = load_and_preprocess_data(args)

    # Set up arguments for transfer learning if required
    if args.transfer_learning != "none":
        set_args_for_transfer_learning(args, pretrained_seed, hyperparameters)

    # Set up the results directory
    results_dir, losses_dir, models_dir = set_results_dir(args.dataset, args.target)
    
    # Set up the file name
    file_name = set_file_name(args)

    # Train the model
    train_model(args, device, dataset, custom_tree, file_name, results_dir, losses_dir, models_dir)