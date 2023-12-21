import argparse

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TreeModel
from pipeline import Pipeline


class EmbeddingPipeline(Pipeline):
    def __init__(self, seed):
        super().__init__(seed)
        self.reducers = {"pca": PCA(random_state=seed), "tsne": TSNE(random_state=seed), "umap": umap.UMAP(random_state=seed)}
        self.taxonomic_ranks = ["Life", "Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species", "Strain"]
        
    def _validate_model(self):
        if not isinstance(self.model, TreeModel):
            raise ValueError("The model must be a TreeModel.")

    def _fetch_node_outputs(self, depth):
        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size=2048, shuffle=False)
        embeddings = []

        for inputs, _ in dataloader:
            self.model(inputs)
            node_outputs = [self.model.outputs[ete_node.name][0].detach().cpu().numpy() 
                            for ete_node in self.tree.ete_tree.traverse("levelorder") 
                            if self.tree.depths[ete_node.name] == depth and ete_node.name in self.model.outputs]

            if node_outputs:
                embeddings.append(np.concatenate(node_outputs, axis=1))
            else:
                raise ValueError(f"No nodes found at depth {depth}.")

        return np.concatenate(embeddings, axis=0)

    def _extract_embeddings_at_depth(self, depth):
        depth = min(depth, self.tree.max_depth)
        return self._fetch_node_outputs(depth)

    def _plot_embeddings_with_labels(self, X_transformed, labels, title, ax):
        for label in labels:
            idx = np.where(self.dataset.y == label)
            ax.scatter(*X_transformed[idx].T[:2], label=label, alpha=0.6)

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2" if X_transformed.shape[1] > 1 else "Constant")
        ax.set_title(title)
        ax.legend()

    def _visualize_embeddings_across_depths(self, reducer):
        depths = range(1, self.tree.max_depth + 1)
        labels = np.unique(self.dataset.y)
        n_plots = len(depths)
        n_cols, n_rows = (n_plots + 1) // 2, 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        for i, depth in tqdm(enumerate(depths, start=1), total=n_plots):
            embeddings = self._extract_embeddings_at_depth(depth)
            X_transformed = self.reducers[reducer].fit_transform(embeddings)
            title = f"{reducer.upper()} "
            title += f"({self.taxonomic_ranks[depth]})" if depth < len(self.taxonomic_ranks) else f"(Raw Data)"
            self._plot_embeddings_with_labels(X_transformed, labels, title, axes.flatten()[i - 1])

        for ax in axes.flatten()[n_plots:]:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/embeddings/{reducer}.png")
        plt.show()

    def run(self, dataset, target, model_fn, reducer):
        self._load_data_and_tree(dataset, target)
        model_fp = f"{self.output_dir}/models/{model_fn}.pt"
        results_fp = f"{self.output_dir}/results/{model_fn.replace('model', 'result')}.json"
        self._load_model(model_fp, results_fp)
        self._validate_model()
        self._visualize_embeddings_across_depths(reducer)

def run_embeddings_pipeline(dataset, target, model_fn, reducer, seed):
    pipeline = EmbeddingPipeline(seed)
    pipeline.run(dataset, target, model_fn, reducer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument("--target", type=str, required=True, help="Target to predict.")
    parser.add_argument("--model_fn", type=str, required=True, help="Filename of the model to load.")
    parser.add_argument("--reducer", type=str, required=True, choices=["pca", "tsne", "umap"], help="Dimensionality reduction technique to use.")
    args = parser.parse_args()

    run_embeddings_pipeline(args.dataset, args.target, args.model_fn, args.reducer, args.seed)