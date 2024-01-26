import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MIOSTONEModel
from pipeline import Pipeline


class EmbeddingPipeline(Pipeline):
    def __init__(self, seed):
        super().__init__(seed)
        self.reducers = {"pca": PCA(random_state=seed), "tsne": TSNE(random_state=seed), "umap": umap.UMAP(random_state=seed)}

    def _create_output_subdir(self):
        self.embedding_dir = self.output_dir + 'embeddings/'
        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)

    def _capture_embeddings(self):
        # Ensure that the model and data are properly set up
        if not self.model or not self.data:
            raise RuntimeError("Model and data must be loaded before capturing embeddings.")
        
        # Validate that the model is an instance of MIOSTONEModel
        if not isinstance(self.model, MIOSTONEModel):
            raise ValueError("Model must be an instance of MIOSTONEModel.")

        # Register hooks to capture embeddings from each layer
        self._register_hooks()

        # Initialize the dictionary of embeddings with the input data
        self.embeddings = {self.tree.max_depth: self.data.X}

        # Perform a forward pass through the model to capture embeddings
        self.model.eval()
        dataloader = DataLoader(self.data, batch_size=2048, shuffle=False)
        for inputs, _ in dataloader:
            self.model(inputs)

        # Unregister hooks to prevent any potential memory leak
        for layer in self.model.hidden_layers:
            layer._forward_hooks.clear()

    def _register_hooks(self):
        def hook_function(module, input, output, depth):
            self.embeddings[depth] = output.detach()

        for depth, layer in enumerate(self.model.hidden_layers):
            layer.register_forward_hook(
                lambda module, input, output, depth=depth: hook_function(module, input, output, depth)
            )

    def _visualize_embeddings_across_depths(self, reducer):
        depths = range(1, self.tree.max_depth + 1)
        labels = np.unique(self.data.y)
        n_plots = len(depths)
        n_cols, n_rows = (n_plots + 1) // 2, 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        for i, depth in tqdm(enumerate(depths, start=1), total=n_plots):
            embeddings = self.embeddings[depth]
            reduced_embeddings = self.reducers[reducer].fit_transform(embeddings)
            title = f"{reducer.upper()} "
            title += f"({self.tree.taxonomic_ranks[depth]})"
            self._plot_embeddings_with_labels(reduced_embeddings, labels, title, axes.flatten()[i - 1])

        for ax in axes.flatten()[n_plots:]:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.embedding_dir}/{reducer}.png")
        plt.show()

    def _plot_embeddings_with_labels(self, reduced_embeddings, labels, title, ax):
        for label in labels:
            idx = np.where(self.data.y == label)
            ax.scatter(*reduced_embeddings[idx].T[:2], label=label, alpha=0.6)

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2" if reduced_embeddings.shape[1] > 1 else "Constant")

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(title)
        ax.legend()

    def run(self, dataset, target, model_fn, reducer, *args, **kwargs):
        self.filepaths = {
            'data': f'../data/{dataset}/data.tsv.xz',
            'meta': f'../data/{dataset}/meta.tsv',
            'target': f'../data/{dataset}/{target}.py',
            'tree': '../data/WoL2/taxonomy.nwk'
        }
        self._load_tree(self.filepaths['tree'])
        self._load_data(self.filepaths['data'], self.filepaths['meta'], self.filepaths['target'], preprocess=True)
        self._create_output_subdir()
        model_fp = f"{self.output_dir}/models/{model_fn}.pt"
        results_fp = f"{self.output_dir}/predictions/{model_fn}.json"
        self._load_model(model_fp, results_fp)
        self._capture_embeddings()
        self._visualize_embeddings_across_depths(reducer)

def run_embedding_pipeline(dataset, target, model_fn, reducer, seed=42):
    pipeline = EmbeddingPipeline(seed)
    pipeline.run(dataset, target, model_fn, reducer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument("--target", type=str, required=True, help="Target to predict.")
    parser.add_argument("--model_fn", type=str, required=True, help="Filename of the model to load.")
    parser.add_argument("--reducer", type=str, default='pca', choices=["pca", "tsne", "umap"], help="Dimensionality reduction technique to use.")
    args = parser.parse_args()

    run_embedding_pipeline(args.dataset, args.target, args.model_fn, args.reducer, args.seed)