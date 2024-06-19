import argparse
import itertools
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import (DeepLift, GradientShap, IntegratedGradients,
                         LayerDeepLift, LayerGradientShap,
                         LayerIntegratedGradients)
from d3blocks import D3Blocks
from ete4.treeview import RectFace, TreeStyle
from matplotlib import cm
from matplotlib.colors import to_hex
from path_explain import PathExplainerTorch
from tqdm import tqdm

from model import MIOSTONEModel
from pipeline import Pipeline


class AttributionPipeline(Pipeline):
    def __init__(self, seed, explainer, depth):
        super().__init__(seed)
        self.explainer = explainer
        self.depth = depth
        self.internal_attributions = {}
        self.leaf_attributions = {}
        self.ancom_results = None   
        self.leaf_interactions = None
        self.internal_interactions = None

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

        inputs = torch.tensor(np.vstack(inputs_list)).requires_grad_()
        baselines = torch.tensor(np.vstack(baselines_list))
        targets = torch.tensor(targets_list)

        return inputs, baselines, targets

    def _normalize_attributions(self, attributions):
        min_value = min(attributions.values())
        max_value = max(attributions.values())
        for node, value in attributions.items():
            attributions[node] = (value - min_value) / (max_value - min_value)
        

    def _load_ancom_results(self, results_path):
        self.ancom_results = pd.read_csv(results_path)
        self.ancom_results.set_index('taxon', inplace=True)

    def _load_correlations(self, correlations_path, p_values_path):
        self.correlations = pd.read_csv(correlations_path, index_col=0)
        self.p_values = pd.read_csv(p_values_path, index_col=0)

    def _compute_internal_attributions(self, inputs, baselines, targets):
        if not self.model:
            raise RuntimeError("Model must be loaded before computing attributions.")
        if not isinstance(self.model, MIOSTONEModel):
            raise ValueError("Model must be an instance of MIOSTONEModel.")
        if self.depth < 0 or self.depth >= self.tree.max_depth:
            raise ValueError(f"Depth must be between 0 and {self.tree.max_depth - 1}.")
        
        layer = self.model.hidden_layers[self.depth]
        if self.explainer == "deeplift":
            attributor = LayerDeepLift(self.model, layer, multiply_by_inputs=False)
        elif self.explainer == "integrated":
            attributor = LayerIntegratedGradients(self.model, layer, multiply_by_inputs=False)
        elif self.explainer == "gradshap":
            attributor = LayerGradientShap(self.model, layer, multiply_by_inputs=False)
        else:
            raise ValueError(f"Invalid explainer: {self.explainer}")

        attribution = attributor.attribute(inputs, baselines=baselines, target=targets).detach().cpu().numpy()
        for node_name, indices in layer.connections.items():
            output_indices = indices[1]
            self.internal_attributions[node_name] = np.mean(np.sum(np.abs(attribution[:, output_indices]), axis=1), axis=0).astype(float)

        self._normalize_attributions(self.internal_attributions)

    def compute_R2(self, inputs, baseline, targets):
        # Ensure valid depth
        if self.depth < 0 or self.depth >= len(self.model.hidden_layers):
            raise ValueError("Invalid layer depth provided.")

        layer = self.model.hidden_layers[self.depth]
        attributors = {
            'DeepLIFT': LayerDeepLift(self.model, layer, multiply_by_inputs=False),
            'Integrated Gradients': LayerIntegratedGradients(self.model, layer, multiply_by_inputs=False),
            'GradientShap': LayerGradientShap(self.model, layer, multiply_by_inputs=False)
        }

        # Compute attributions for each method
        attributions = {}
        for name, attributor in attributors.items():
            attr = attributor.attribute(inputs, baselines=baseline, target=targets).detach().cpu().numpy()
            attributions[name] = attr

        # Summarize node-level attributions
        internal_attributions = {name: {} for name in attributors.keys()}
        for node_name, (_, output_indices) in layer.connections.items():
            for name, attr in attributions.items():
                node_attribution = np.mean(np.sum(np.abs(attr[:, output_indices]), axis=1), axis=0)
                internal_attributions[name][node_name] = node_attribution.astype(float)

        # Normalize attributions
        for name, attribs in internal_attributions.items():
            self._normalize_attributions(attribs)

        # Prepare DataFrame for visualization
        df = pd.DataFrame(internal_attributions)

        # Setup pairplot with regression line and R² annotations
        g = sns.pairplot(df, kind='scatter', diag_kind=None)
        for i in range(len(df.columns)):
            for j in range(len(df.columns)):
                r2 = np.corrcoef(df.iloc[:, j], df.iloc[:, i])[0, 1] ** 2  # Calculate R^2 for each pair
                g.axes[i, j].annotate(f'R² = {r2:.4f}', (0.5, 0.8), xycoords='axes fraction', ha='center', va='center', fontsize=9)
                # Adding y=x line
                g.axes[i, j].plot([0, 1], [0, 1], 'k--', alpha=0.75, zorder=0)  # Plot y=x line

        # Set uniform ticks for both x and y axes
        tick_labels = np.round(np.linspace(df.min().min(), df.max().max(), num=5), 2)
        for ax in g.axes.flatten():
            ax.set_xticks(tick_labels)
            ax.set_yticks(tick_labels)

        plt.show()

    def _compute_leaf_attributions(self, inputs, baselines, targets):
        if self.explainer == "deeplift":
            attributor = DeepLift(self.model, multiply_by_inputs=False)
        elif self.explainer == "integrated":
            attributor = IntegratedGradients(self.model, multiply_by_inputs=False)
        elif self.explainer == "gradshap":
            attributor = GradientShap(self.model, multiply_by_inputs=False)
        else:
            raise ValueError(f"Invalid explainer: {self.explainer}")
        attributions = attributor.attribute(inputs, baselines=baselines, target=targets).detach().cpu().numpy()
        for i, feature in enumerate(self.data.features):
            self.leaf_attributions[feature] = np.mean(np.abs(attributions), axis=0)[i].astype(float)

        self._normalize_attributions(self.leaf_attributions)

    def _compute_leaf_interactions(self, inputs, baselines, targets):
        self.leaf_interactions = pd.DataFrame(columns=self.data.features, index=self.data.features)

        if os.path.exists(f"{self.attributions_dir}/leaf_interactions.csv"):
            print("Loading pre-computed leaf interactions...")
            interactions_df = pd.read_csv(f"{self.attributions_dir}/leaf_interactions.csv", index_col=0)
            self.leaf_interactions = interactions_df
            return
       
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            baselines = baselines.cuda()
            targets = targets.cuda()
            self.model.cuda()
        
        attributor = PathExplainerTorch(self.model)
        for idx in range(len(self.data.features)):
            print(f"Computing interactions for feature {idx + 1}/{len(self.data.features)} ({self.data.features[idx]})")
            interactions = attributor.interactions(inputs, baseline=baselines[0], use_expectation=False, output_indices=targets, interaction_index=idx).detach().cpu().numpy()
            interactions = np.mean(np.abs(interactions), axis=0)
            for leaf_idx, leaf_name in enumerate(self.data.features):
                self.leaf_interactions[self.data.features[idx]][leaf_name] = interactions[leaf_idx]
        
        # Save leaf interactions to file
        print("Saving leaf interactions to file...")
        self.leaf_interactions.to_csv(f"{self.attributions_dir}/leaf_interactions.csv")
    
    def _retrieve_interactions(self, node1, node2):
        # Retrieve pre-computed interactions between descendants
        descendants1 = list(node1.leaf_names())
        descendants2 = list(node2.leaf_names())
        interactions = self.leaf_interactions.loc[descendants1, descendants2].values

        return interactions
        
    def _compute_internal_interactions(self, method="sum"):
        filepath= f"{self.attributions_dir}/internal_interactions_{self.depth}_{method}.csv"
        if os.path.exists(filepath):
            interactions_df = pd.read_csv(filepath, index_col=0)
            self.internal_interactions = interactions_df
            return
        
        nodes = {ete_node: ete_node for ete_node in self.tree.ete_tree.traverse("levelorder") if self.tree.depths[ete_node.name] == self.depth}
        node_names = [node.name for node in nodes.values()]
        self.internal_interactions = pd.DataFrame(columns=node_names, index=node_names)
        for node1, node2 in tqdm(itertools.combinations_with_replacement(nodes.values(), 2), total=len(nodes) * (len(nodes) + 1) // 2):
            if node1 == node2:
                self.internal_interactions.loc[node1.name, node2.name] = 0
                continue
            interactions = self._retrieve_interactions(node1, node2)
            if method == "mean":
                interaction_value = np.mean(interactions)
            elif method == "max":
                interaction_value = np.max(interactions)
            elif method == "sum":
                interaction_value = np.sum(interactions)
            self.internal_interactions.loc[node1.name, node2.name] = interaction_value

        # Normalize internal interactions
        self.internal_interactions = self.internal_interactions.astype(float)
        min_value = self.internal_interactions.min().min()
        max_value = self.internal_interactions.max().max()
        self.internal_interactions = (self.internal_interactions - min_value) / (max_value - min_value)

        # Save internal interactions to file
        self.internal_interactions.to_csv(filepath)


    def _generate_colors(self, attributions):
        colormap = sns.color_palette("YlGn_d", as_cmap=True)
        colors = {node: to_hex(colormap(value)) for node, value in attributions.items()}

        # Generate a color bar for the colormap
        if "colorbar.pdf" not in os.listdir(self.attributions_dir):
            fig, ax = plt.subplots(figsize=(6, 1))
            fig.subplots_adjust(bottom=0.5)
            norm = plt.Normalize(0, 1)
            cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap),
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

    def _visualize_attributions(self):
        # Print the top 10 internal nodes with the highest DeepLIFT attributions
        print("Top 10 internal nodes with the highest DeepLIFT attributions:")
        for node_name, value in sorted(self.internal_attributions.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{node_name}: {value}")
            '''
            i = 0
            for leaf_name, leaf_value in sorted(self.leaf_attributions.items(), key=lambda x: x[1], reverse=True):
                leaf = list(self.tree.ete_tree.search_leaves_by_name(name=leaf_name))[0]
                if list(leaf.search_ancestors(name=node_name)) != []:
                    print(f"\t{leaf_name}: {leaf_value}")
                    print(f"\t{leaf.up.name}")
                    i += 1
                if i == 2:  
                    break
            '''
            # Print the node's ancestors
            node = list(self.tree.ete_tree.search_nodes(name=node_name))[0]
            for ancestor in node.ancestors():
                print(f"\t{ancestor.name}")

        # Generate colors
        internal_colors = self._generate_colors(self.internal_attributions)
        # leaf_colors = self._generate_colors(self.leaf_attributions)

        # Configure tree style
        ts = TreeStyle()
        ts.mode = "c"
        ts.show_leaf_name = False
        ts.show_scale = False

        # Configure colors
        self._configure_internal_colors(internal_colors)
        # self._configure_leaf_colors(leaf_colors)

        # Render tree
        image_path = f"{self.attributions_dir}/tree_{self.depth}_{self.explainer}.png"
        self.tree.ete_tree.show(tree_style=ts)
        # self.tree.ete_tree.render(image_path, tree_style=ts, h=2400, units="px", dpi=1200)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(plt.imread(image_path))
        plt.show()
        

    def _visualize_interactions(self):
        # Convert interactions to a DataFrame suitable for D3 Chord visualization
        interactions = self.internal_interactions.stack().reset_index()
        interactions.columns = ["source", "target", "weight"]
        interactions["source"] = interactions["source"].apply(lambda x: x.split("__")[1])
        interactions["target"] = interactions["target"].apply(lambda x: x.split("__")[1])

        # Remove self-interactions
        interactions = interactions[interactions["source"] != interactions["target"]]

        # Map node names to node objects while filtering by depth
        name2node = {node.name.split("__")[1]: node for node in self.tree.ete_tree.traverse("levelorder") if self.tree.depths[node.name] == self.depth}

        # Classifications for filtering and coloring
        classes = ['c__Bacilli', 'c__Gammaproteobacteria', 'c__Clostridia', 'c__Bacteroidia', 'c__Actinomycetia']
        node2class = {}

        for i, row in interactions.iterrows():
            for node_key in ['source', 'target']:
                for ancestor in name2node[row[node_key]].ancestors():
                    if ancestor.name in classes:
                        node2class[row[node_key]] = ancestor.name
                        break
        
        # Filter interactions based on class presence
        interactions = interactions[interactions["source"].isin(node2class) & interactions["target"].isin(node2class)]

        # Remove interactions that are less than the 90th percentile
        weight_threshold = interactions['weight'].quantile(0.99)
        top_interactions = interactions[interactions['weight'] >= weight_threshold]
        print(top_interactions["weight"].describe())

        # Define class colors
        class_colors = {'c__Bacilli': '#d62728', 'c__Gammaproteobacteria': '#9467bd', 'c__Clostridia': '#8c564b', 'c__Bacteroidia': '#e377c2', 'c__Actinomycetia': '#7f7f7f'}

        # Order nodes by class and name
        ordering = list(set(top_interactions['source']) | set(top_interactions['target']))
        ordering.sort(key=lambda x: (node2class.get(x, ""), x))

        # Initialize D3 visualization
        d3 = D3Blocks()
        d3.chord(df=top_interactions, arrowhead=-1, showfig=False, ordering=ordering)

        # Assign colors to nodes based on class
        d3.node_properties['color'] = d3.node_properties['label'].map(node2class).map(class_colors)

        # Assign orange color to edges
        d3.edge_properties['color'] = "#ff7f0e"

        d3.show()

        return interactions, node2class

    def _visualize_correlations(self, interactions, node2class):
        correlations = self.correlations.stack().reset_index()
        correlations.columns = ["source", "target", "weight"]
        p_values = self.p_values.stack().reset_index()
        p_values.columns = ["source", "target", "p_value"]
        interactions.columns = ["source", "target", "interaction"]
        
        # Merge correlations and p-values
        correlations = pd.merge(correlations, p_values, on=["source", "target"])
        correlations["source"] = correlations["source"].apply(lambda x: x.split("__")[1])
        correlations["target"] = correlations["target"].apply(lambda x: x.split("__")[1])

        # Merge correlations and interactions
        correlations = pd.merge(interactions, correlations, on=["source", "target"], how="left")
        
        # Use absolute values for weights
        correlations["weight"] = correlations["weight"].abs()

        # Plot weight vs. interaction
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=correlations, x="weight", y="interaction", hue="p_value", palette="viridis")
        plt.xlabel("Correlation")
        plt.ylabel("Interaction")
        plt.show()
        
        # Remove correlations that are less than the 99th percentile
        weight_threshold = correlations['weight'].quantile(0.99)
        top_correlations = correlations[correlations['weight'] >= weight_threshold]

        # Set weight to be the difference between the interaction weight and the correlation weight
        correlations["weight"] = correlations["interaction"] - correlations["weight"]
        correlations["weight"] = correlations["weight"].apply(lambda x: max(0, x))

        # Define class colors
        class_colors = {'c__Bacilli': '#d62728', 'c__Gammaproteobacteria': '#9467bd', 'c__Clostridia': '#8c564b', 'c__Bacteroidia': '#e377c2', 'c__Actinomycetia': '#7f7f7f'}

        # Order nodes by class and name
        ordering = list(set(top_correlations['source']) | set(top_correlations['target']))
        ordering.sort(key=lambda x: (node2class.get(x, ""), x))

        # Initialize D3 visualization
        d3 = D3Blocks()
        d3.chord(df=top_correlations, arrowhead=-1, showfig=False, ordering=ordering)

        # Assign colors to nodes based on class
        d3.node_properties['color'] = d3.node_properties['label'].map(node2class).map(class_colors)

        # Assign orange color to edges
        d3.edge_properties['color'] = "#ff7f0e"

        d3.show()


    def visualize_ancom(self):
        # Extracting log fold change and p-values from ANCOM results
        log_fold_changes = self.ancom_results['lfc_diagnosisuc']
        p_values = self.ancom_results['p_diagnosisuc']

        # Creating a DataFrame for easier plotting
        data = pd.DataFrame({
            'Taxon': self.ancom_results.index,
            'Log Fold Change': log_fold_changes,
            'NegLog10 P-Value': -np.log10(p_values)
        })

        # Adding importance values
        data['Feature Importance Values'] = data['Taxon'].apply(lambda x: self.internal_attributions.get(x, 0) ** 2)

        # Sorting data by 'Feature Importance Values' to annotate the top 5 taxa
        sorted_data = data.sort_values(by='Feature Importance Values', ascending=True)
        top_data = sorted_data.tail(10)

        # Plotting using Seaborn
        sns.set_theme(style='white')
        plt.figure(figsize=(15, 15))
        palette = sns.color_palette('YlGn_d', as_cmap=True)
        scatter_plot = sns.scatterplot(data=sorted_data,
                                    x='Log Fold Change', 
                                    y='NegLog10 P-Value', 
                                    hue='Feature Importance Values', 
                                    size='Feature Importance Values', 
                                    sizes=(10, 200),
                                    palette=palette,
                                    alpha=0.6)
        

        # Annotating the top taxa with their ranks
        i = 10
        for _, row in top_data.iterrows():
            scatter_plot.text(row['Log Fold Change'], row['NegLog10 P-Value'], i, fontsize=8)
            i -= 1

        # Adding horizontal lines for significant p-value thresholds
        line05 = plt.axhline(-np.log10(0.05), color='grey', linestyle='dotted')
        line01 = plt.axhline(-np.log10(0.01), color='grey', linestyle='dashed')

        # Enhancing the plot with titles, labels
        plt.xlabel('Log Fold Change')
        plt.ylabel('-log10(P-Value)')

        # Add legend for feature importance values
        handles, labels = scatter_plot.get_legend_handles_labels()
        handles.extend([line05, line01])
        labels.extend(['P-Value = 0.05', 'P-Value = 0.01'])
        scatter_plot.legend(handles=handles, labels=labels, loc='upper right')

        plt.show()

    def run(self, dataset, target, model_fn, num_samples, *args, **kwargs):
        # Define filepaths
        self.filepaths = {
            'data': f'../data/{dataset}/data.tsv.xz',
            'meta': f'../data/{dataset}/meta.tsv',
            'target': f'../data/{dataset}/{target}.py',
            'tree': '../data/WoL2/taxonomy.nwk',
            "model": f"../output/{dataset}/{target}/models/{model_fn}.pt",
            "results": f"../output/{dataset}/{target}/predictions/{model_fn}.json",
            "ancom": f"../output/{dataset}/{target}/ancom/ancom_{self.depth}.csv",
            "correlations": f"../output/{dataset}/{target}/attributions/correlations_{self.depth}.csv",
            "p_values": f"../output/{dataset}/{target}/attributions/p_values_{self.depth}.csv"
        }
        self._load_tree(self.filepaths['tree'])
        self._load_data(self.filepaths['data'], self.filepaths['meta'], self.filepaths['target'])
        self._create_output_subdir()
        self._load_ancom_results(self.filepaths['ancom'])
        self._load_correlations(self.filepaths['correlations'], self.filepaths['p_values'])
        self._load_model(self.filepaths['model'], self.filepaths['results'])
        num_samples = len(self.data) if num_samples is None else num_samples
        inputs, baselines, targets = self._preprocess_data(num_samples)
        # self._compute_internal_attributions(inputs, baselines, targets)
        # self._compute_leaf_attributions(inputs, baselines, targets)
        # self.compute_R2(inputs, baselines, targets)
        self._compute_leaf_interactions(inputs, baselines, targets)
        self._compute_internal_interactions()
        # self._visualize_attributions()
        interactions, node2class = self._visualize_interactions()
        self._visualize_correlations(interactions, node2class)
        # self.visualize_ancom()


def run_attribution_pipeline(dataset, target, model_fn, explainer, depth, num_samples=None, seed=42):
    pipeline = AttributionPipeline(seed, explainer, depth)
    pipeline.run(dataset, target, model_fn, num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument("--target", type=str, required=True, help="Target to predict.")
    parser.add_argument("--model_fn", type=str, required=True, help="Model filename to use.")
    parser.add_argument("--explainer", type=str, default="deeplift", help="Explainer to use for attribution.")
    parser.add_argument("--depth", type=int, default=5, help="Depth to visualize for internal attributions.")
    parser.add_argument("--num_samples", type=int, help="Number of samples to use for attribution. If not specified, use all samples.")
    args = parser.parse_args()

    run_attribution_pipeline(args.dataset, args.target, args.model_fn, args.explainer, args.depth, args.num_samples, args.seed)