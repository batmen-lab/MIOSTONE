import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import DeepLift, LayerDeepLift
from ete4.treeview import RectFace, TreeStyle
from matplotlib import cm
from matplotlib.colors import to_hex

from model import MIOSTONEModel
from pipeline import Pipeline


class AttributionPipeline(Pipeline):
    def __init__(self, seed):
        super().__init__(seed)
        self.internal_attributions = {}
        self.leaf_attributions = {}      

    def _create_output_subdir(self):
        super()._create_output_subdir()
        self.attributions_dir = self.output_dir + 'attributions/'
        if not os.path.exists(self.attributions_dir):
            os.makedirs(self.attributions_dir)  

    def _preprocess_data(self, num_samples):
        selected_indices = np.random.choice(len(self.data), num_samples, replace=False)
        inputs = self.data.X[selected_indices]
        targets = self.data.y[selected_indices]

        selected_indices = np.random.choice(len(self.data.y), num_samples, replace=False)
        inputs_list = self.data.X[selected_indices]
        baselines_list = np.mean(self.data.X, axis=0)
        baselines_list = np.tile(baselines_list, (num_samples, 1))
        targets_list = self.data.y[selected_indices]

        inputs = torch.tensor(np.vstack(inputs_list))
        baselines = torch.tensor(np.vstack(baselines_list))
        targets = torch.tensor(targets_list)

        return inputs, baselines, targets
    
    def _normalize_attributions(self, attributions):
        min_value = min(attributions.values())
        max_value = max(attributions.values())
        for node, value in attributions.items():
            attributions[node] = (value - min_value) / (max_value - min_value)

    def _compute_internal_attributions(self, inputs, baselines, targets, depth):
        if not self.model:
            raise RuntimeError("Model must be loaded before computing attributions.")
        if not isinstance(self.model, MIOSTONEModel):
            raise ValueError("Model must be an instance of MIOSTONEModel.")
        if depth < 0 or depth >= self.tree.max_depth:
            raise ValueError(f"Depth must be between 0 and {self.tree.max_depth - 1}.")
        
        layer = self.model.hidden_layers[depth]
        attributor = LayerDeepLift(self.model, layer, multiply_by_inputs=False)
        attribution = attributor.attribute(inputs, baselines=baselines, target=targets).detach().cpu().numpy()
        for node_name, indices in layer.connections.items():
            output_indices = indices[1]
            self.internal_attributions[node_name] = np.mean(np.sum(np.abs(attribution[:, output_indices]), axis=1), axis=0).astype(float)

        self._normalize_attributions(self.internal_attributions)
        
    def _compute_leaf_attributions(self, inputs, baselines, targets):
        attributor = DeepLift(self.model)
        attributions = attributor.attribute(inputs, baselines=baselines, target=targets).detach().cpu().numpy()
        for i, feature in enumerate(self.data.features):
            self.leaf_attributions[feature] = np.mean(np.abs(attributions), axis=0)[i].astype(float)

        self._normalize_attributions(self.leaf_attributions)

    def _generate_colors(self, attributions):
        colormap = cm.YlGn
        colors = {node: to_hex(colormap(value)) for node, value in attributions.items()}

        # Generate a color bar for the colormap
        if "colorbar.pdf" not in os.listdir(self.attributions_dir):
            fig, ax = plt.subplots(figsize=(6, 1))
            fig.subplots_adjust(bottom=0.5)
            cmap = cm.YlGn
            norm = plt.Normalize(0, 1)
            cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                                cax=ax,
                                orientation='horizontal')
            cb1.set_label('Feature Importance Range')
            plt.savefig(f"{self.attributions_dir}/colorbar.pdf")
            plt.show()
        return colors

    def _configure_internal_colors(self, internal_colors):
        for ete_node in self.tree.ete_tree.traverse("levelorder"):
            if ete_node.name in internal_colors:
                for descendant in [ete_node] + list(ete_node.descendants()):
                    descendant.img_style["fgcolor"] = internal_colors[ete_node.name]
                    descendant.img_style["hz_line_color"] = internal_colors[ete_node.name]
                    descendant.img_style["vt_line_color"] = internal_colors[ete_node.name]
                    descendant.img_style["hz_line_type"] = 0
                    descendant.img_style["vt_line_type"] = 0

    def _configure_leaf_colors(self, leaf_colors):
        white_rectface = RectFace(width=100, height=5, fgcolor="white", bgcolor="white")
        for leaf in self.tree.ete_tree.leaves():
            leaf.add_face(white_rectface, column=0, position="aligned")
            leaf.add_face(white_rectface, column=1, position="aligned")
            color = leaf_colors.get(leaf.name, "white") 
            colored_rectface = RectFace(width=200, height=5, fgcolor=color, bgcolor=color)
            leaf.add_face(colored_rectface, column=2, position="aligned")

    def _visualize(self):
        # Print the top 10 internal nodes with the highest DeepLIFT attributions
        print("Top 10 internal nodes with the highest DeepLIFT attributions:")
        for node_name, value in sorted(self.internal_attributions.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{node_name}: {value}")
            i = 0
            for leaf_name, leaf_value in sorted(self.leaf_attributions.items(), key=lambda x: x[1], reverse=True):
                leaf = list(self.tree.ete_tree.search_leaves_by_name(name=leaf_name))[0]
                if list(leaf.search_ancestors(name=node_name)) != []:
                    print(f"\t{leaf_name}: {leaf_value}")
                    print(f"\t{leaf.up.name}")
                    i += 1
                if i == 2:
                    break

        # Generate colors
        internal_colors = self._generate_colors(self.internal_attributions)
        leaf_colors = self._generate_colors(self.leaf_attributions)

        # Configure tree style
        ts = TreeStyle()
        ts.mode = "c"
        ts.show_leaf_name = False
        ts.show_scale = False

        # Configure colors
        self._configure_internal_colors(internal_colors)
        self._configure_leaf_colors(leaf_colors)

        # Render tree
        image_path = f"{self.attributions_dir}/tree.png"
        self.tree.ete_tree.render(image_path, tree_style=ts, h=2400, units="px", dpi=1200)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(plt.imread(image_path))
        plt.show()

    def run(self, dataset, target, model_fn, depth, num_samples, *args, **kwargs):
        # Define filepaths
        self.filepaths = {
            'data': f'../data/{dataset}/data.tsv.xz',
            'meta': f'../data/{dataset}/meta.tsv',
            'target': f'../data/{dataset}/{target}.py',
            'tree': '../data/WoL2/taxonomy.nwk',
            "model": f"../output/{dataset}/{target}/models/{model_fn}.pt",
            "results": f"../output/{dataset}/{target}/predictions/{model_fn}.json"
        }
        self._load_tree(self.filepaths['tree'])
        self._load_data(self.filepaths['data'], self.filepaths['meta'], self.filepaths['target'])
        self._create_output_subdir()
        self._load_model(self.filepaths['model'], self.filepaths['results'])
        num_samples = len(self.data) if num_samples is None else num_samples
        inputs, baselines, targets = self._preprocess_data(num_samples)
        self._compute_internal_attributions(inputs, baselines, targets, depth)
        self._compute_leaf_attributions(inputs, baselines, targets)
        self._visualize()

def run_attribution_pipeline(dataset, target, model_fn, depth, num_samples=None, seed=42):
    pipeline = AttributionPipeline(seed)
    pipeline.run(dataset, target, model_fn, depth, num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument("--target", type=str, required=True, help="Target to predict.")
    parser.add_argument("--model_fn", type=str, required=True, help="Model filename to use.")
    parser.add_argument("--depth", type=int, default=4, help="Depth to visualize for internal attributions.")
    parser.add_argument("--num_samples", type=int, help="Number of samples to use for attribution. If not specified, use all samples.")
    args = parser.parse_args()

    run_attribution_pipeline(args.dataset, args.target, args.model_fn, args.depth, args.num_samples, args.seed)