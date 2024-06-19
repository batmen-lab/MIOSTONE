import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        
        # Validate that the model is an instance of mlp, popphy, taxonn, or miostone
        if self.model_type not in ['mlp', 'popphycnn', 'taxonn', 'miostone']:
            raise ValueError(f"Invalid model type: {self.model_type}")

        # Register hooks to capture embeddings from each layer
        self._register_hooks()

        # Initialize the dictionary of embeddings with the input data
        if self.model_type == 'miostone':
            self.data.clr_transform()
            self.embeddings = {self.tree.max_depth: self.data.X}
        elif self.model_type in ['mlp', 'taxonn']:
            self.data.clr_transform()
            self.embeddings = {1: self.data.X}
        elif self.model_type == 'popphycnn':
            self.data.to_popphycnn_matrix(self.tree)
            self.embeddings = {2: self.data.X}

        # Perform a forward pass through the model to capture embeddings            
        self.model.eval()
        dataloader = DataLoader(self.data, batch_size=2048, shuffle=False)
        for inputs, _ in dataloader:
            self.model(inputs)

        # Unregister hooks to prevent any potential memory leak
        if self.model_type == 'miostone':
            for layer in self.model.hidden_layers:
                layer._forward_hooks.clear()
        elif self.model_type == 'mlp':
            self.model.fc1._forward_hooks.clear()
        elif self.model_type == 'popphycnn':
            self.model.cnn_layers._forward_hooks.clear()
            self.model.fc_layers._forward_hooks.clear()
        elif self.model_type == 'taxonn':
            self.model.output_layer._forward_hooks.clear()


    def _register_hooks(self):
        def hook_function(module, input, output, depth):
            self.embeddings[depth] = output.detach() if self.model_type != 'taxonn' else input[0].detach()

        if self.model_type == 'miostone':
            for depth, layer in enumerate(self.model.hidden_layers):
                layer.register_forward_hook(
                    lambda module, input, output, depth=depth: hook_function(module, input, output, depth)
                )
        elif self.model_type == 'mlp':
            self.model.fc1.register_forward_hook(
                lambda module, input, output: hook_function(module, input, output, 0)
            )
        elif self.model_type == 'popphycnn':
            self.model.fc_layers.register_forward_hook(
                lambda module, input, output: hook_function(module, input, output, 0)
            )
            self.model.cnn_layers.register_forward_hook(
                lambda module, input, output: hook_function(module, input, output, 1)
            )
        elif self.model_type == 'taxonn':
            self.model.output_layer.register_forward_hook(
                lambda module, input, output: hook_function(module, input, output, 0)
            )

    def _visualize_embeddings_across_depths(self, reducer):
        if self.model_type == 'miostone':
            depths = range(1, self.tree.max_depth + 1)
            titles = [f"{self.tree.taxonomic_ranks[depth]}" for depth in depths]
        elif self.model_type == 'mlp':
            depths = range(0, 2)
            titles = ['fc1', 'input']
        elif self.model_type == 'popphycnn':
            depths = range(0, 3)
            titles = ['fc', 'cnn', 'input']
        elif self.model_type == 'taxonn':
            depths = range(0, 2)
            titles = ['cnn', 'input']

        labels = np.unique(self.data.y)
        n_plots = len(depths)
        n_cols, n_rows = (n_plots + 1) // 2, 2 

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        for i, depth in tqdm(enumerate(depths, start=1), total=n_plots):
            embeddings = self.embeddings[depth]
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
            reduced_embeddings = self.reducers[reducer].fit_transform(embeddings)
            title = f"{reducer.upper()} "
            title += f"- {titles[i - 1]}"
            self._plot_embeddings_with_labels(reduced_embeddings, labels, axes.flatten()[i - 1])
            score = silhouette_score(embeddings, self.data.y)
            axes.flatten()[i - 1].set_title(f"{title} (Silhouette Score: {score:.2f})")

        for ax in axes.flatten()[n_plots:]:
            ax.axis('off')

        dataset_name = self.filepaths['data'].split('/')[-2]
        if dataset_name == 'ibd200' or dataset_name == 'hmp2':
            label_mapping = {0: 'UC', 1: 'CD'}
        elif dataset_name == 'alzbiom':
            label_mapping = {0: 'Health', 1: 'AD'}
        elif dataset_name == 'asd':
            label_mapping = {0: 'TD', 1: 'Autism'}
        elif dataset_name == 'gd':
            label_mapping = {0: 'Health', 1: 'Disease'}
        elif dataset_name == 'rumc' or dataset_name == 'tbc':
            label_mapping = {0: 'No', 1: 'Yes'}
        fig.legend([label_mapping[label] for label in labels], loc='center', ncol=2)

        plt.tight_layout()
        plt.savefig(f"{self.embedding_dir}/{self.model_type}_{reducer}.png")
        plt.show()

    def _plot_embeddings_with_labels(self, reduced_embeddings, labels, ax):
        for label in labels:
            idx = np.where(self.data.y == label)
            ax.scatter(*reduced_embeddings[idx].T[:2], label=label, alpha=0.6)

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2" if reduced_embeddings.shape[1] > 1 else "Constant")

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # ax.legend()

    def run(self, dataset, target, model_fn, reducer, *args, **kwargs):
        self.filepaths = {
            'data': f'../data/{dataset}/data.tsv.xz',
            'meta': f'../data/{dataset}/meta.tsv',
            'target': f'../data/{dataset}/{target}.py',
            'tree': '../data/WoL2/taxonomy.nwk'
        }
        self._load_tree(self.filepaths['tree'])
        self._load_data(self.filepaths['data'], self.filepaths['meta'], self.filepaths['target'], preprocess=False)
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