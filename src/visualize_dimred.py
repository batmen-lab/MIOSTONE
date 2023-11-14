import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import depth2taxo, load_data, load_model, preprocess_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize datasets using PCA, t-SNE, and UMAP.")
    parser.add_argument("--seed", type=int, default=3, help="Random seed.")
    parser.add_argument("--dataset", type=str, default="IBD200", help="Dataset to visualize.")
    parser.add_argument("--target", type=str, default="diagnosis", help="Target to visualize.")
    parser.add_argument("--algorithm", type=str, default="pca", choices=["pca", "tsne", "umap"], help="Dimensionality reduction algorithm to use.")
    parser.add_argument("--depth", type=int, default=-1, help="Depth of the model embedding to use. If larger than the maximum depth, the raw data will be used. If negative, all depths (including the raw data) will be visualized.")
    parser.add_argument("--embedding_reduction", type=str, default="none", choices=["none", "mean"], help="Reduction method to use for the node embeddings.")
    return parser.parse_args()


def get_embeddings_at_depth(model, dataset, depth, embedding_reduction):
    # If depth is larger than the maximum depth, use the raw data
    if depth > model.ete_tree.max_depth:
        return dataset.X
    
    model.eval()
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
    
    embeddings = []
    for inputs, _ in dataloader:
        model(inputs)
        node_outputs = []
        for ete_node in model.ete_tree.traverse("levelorder"):
            if ete_node.depth == depth:
                node_name = ete_node.name
                if node_name in model.outputs:
                    output = model.outputs[node_name][0].detach().cpu()
                    if embedding_reduction == "mean":
                        output = torch.mean(output, dim=1).unsqueeze(1)
                    node_outputs.append(output)
                else:
                    raise ValueError(f"Node {node_name} at depth {depth} was not found in the outputs.")

        if not node_outputs:
            raise ValueError(f"No nodes found at depth {depth}.")
        
        embedding = torch.cat(node_outputs, dim=1)
        embeddings.append(embedding)
    
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.numpy()

def visualize_all_depths_single_algorithm(seed, model, dataset, algorithm, embedding_reduction):
    max_depth = model.ete_tree.max_depth
    depths = list(range(1, max_depth)) + [max_depth + 1]
    n_plots = len(depths)

    # Calculate number of rows and columns for subplots
    n_rows = 2
    n_cols = (n_plots + 1) // n_rows

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    
    for i, depth in tqdm(enumerate(depths), total=n_plots):
        embeddings = get_embeddings_at_depth(model, dataset, depth, embedding_reduction)
        X_transformed = apply_dimensionality_reduction(seed, embeddings, algorithm)
        title = "PCA" if algorithm == "pca" else "t-SNE" if algorithm == "tsne" else "UMAP"
        title += f" ({depth2taxo(depth)})"
        
        # Convert 2D axes array to 1D and use it for indexing
        ax = axes.flatten()[i]
        visualize_single_algorithm(X_transformed, dataset.le.inverse_transform(dataset.y), title, ax)
        
    # Hide any unused axes
    for ax in axes.flatten()[n_plots:]:
        ax.axis('off')

    # Adjust the layout
    plt.tight_layout()

def apply_dimensionality_reduction(seed, data, method):
    reducers = {
        "pca": PCA(random_state=seed),
        "tsne": TSNE(random_state=seed),
        "umap": umap.UMAP(random_state=seed)
    }
    return reducers[method].fit_transform(data)

def visualize_single_algorithm(X, y, title, ax):
    unique_labels = np.unique(y)
    if X.shape[1] == 1:
        # Handling 1-dimensional data
        for label in unique_labels:
            idx = np.where(y == label)
            ax.scatter(X[idx, 0], np.zeros_like(X[idx, 0]), label=label, alpha=0.6)  # Plotting on a line
    else:
        # Handling 2-dimensional data
        for label in unique_labels:
            idx = np.where(y == label)
            ax.scatter(X[idx, 0], X[idx, 1], label=label, alpha=0.6)

    ax.set_xlabel("Component 1")
    if X.shape[1] > 1:
        ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.legend()

def visualize_embeddings(args, model, dataset):
    # Visualize embeddings and dimensionality reduction
    if args.depth < 0:
        visualize_all_depths_single_algorithm(args.seed, model, dataset, args.algorithm, args.embedding_reduction)
    elif args.depth >= 0:
        dataset.X = get_embeddings_at_depth(model, dataset, args.depth, args.embedding_reduction) 
        X_transformed = apply_dimensionality_reduction(args.seed, dataset.X, args.algorithm)
        title = "PCA" if args.algorithm == "pca" else "t-SNE" if args.algorithm == "tsne" else "UMAP"
        title += f" (Raw Data)" if args.depth >= model.ete_tree.max_depth + 1 else f" (Depth {args.depth})"
        visualize_single_algorithm(X_transformed, dataset.le.inverse_transform(dataset.y), title, plt.gca())

    plt.show()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load model and data
    model_dir = f"../results/{args.dataset}/{args.target}/models"
    model_path = os.path.join(model_dir, os.listdir(model_dir)[0])
    model, _, hyperparams = load_model(model_path)
    
    data, metadata = load_data(f'../data/{args.dataset}', args.dataset)
    _, dataset = preprocess_data(data, metadata, args.dataset, args.target, "WoL2", hyperparams["taxonomy"], 
                                 hyperparams["prob_label"], hyperparams["prune_by_dataset"])
    le = dataset.le
    dataset = dataset.preprocess(hyperparams["preprocess"])
    dataset.le = le
    
    # Visualize embeddings
    visualize_embeddings(args, model, dataset)
    