import argparse
import json
import os

import matplotlib as mpl
import numpy as np
import torch
from captum.attr import DeepLift, LayerDeepLift
from ete3 import RectFace, TreeStyle
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from utils import load_data, load_model, preprocess_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize explanations for the model.")
    parser.add_argument("--seed", type=int, default=3, help="Random seed.")
    parser.add_argument("--dataset", type=str, default="IBD200", help="Dataset name or directory.")
    parser.add_argument("--target", type=str, default="diagnosis", help="Target variable.")
    parser.add_argument("--leaf_attr", type=str, choices=["gini_gain", "pb_correlation", "auroc", "deeplift"], help="Attributions to compute for the leaves.")
    parser.add_argument("--layer_deeplift", action="store_true", help="Compute Layer DeepLift.")
    parser.add_argument("--num_samples", type=int, help="Number of samples to use for computing deepLIFT. If not specified, use all samples.")
    parser.add_argument("--top_k_nodes", type=int, help="Number of top nodes to select at each depth for computing DeepLift. If not specified, use all nodes.")
    parser.add_argument("--min_depth", type=int, help="Minimum depth to compute attributions for. If not specified, 0 is used.")
    parser.add_argument("--max_depth", type=int, help="Maximum depth to compute attributions for. If not specified, the maximum depth of the tree is used.")
    parser.add_argument("--save_attributions", action="store_true", help="Save attributions to file.")
    parser.add_argument("--visualize", action="store_true", help="Visualize attributions.")
    parser.add_argument("--visualize_depth", type=int, default=4, help="Visualize attributions for the specified depth.")
    return parser.parse_args()

def set_attributions_dir(args):
    attributions_dir = f"../results/{args.dataset}/{args.target}/attributions"
    if not os.path.exists(attributions_dir):
        os.makedirs(attributions_dir)
    return attributions_dir

def compute_attributions(args, model_path, model, dataset, attributions_dir):

    # Set default values for num_samples
    if args.num_samples is None:
        args.num_samples = len(dataset)
    if args.min_depth is None:
        args.min_depth = 0

    name2node = {node.name: node for node in model.ete_tree.traverse('levelorder')}

    if args.leaf_attr == "deeplift":
        inputs, baselines, targets = prepare_data_for_deeplift(model, dataset, args.num_samples)
        feature_deeplift_values = compute_deeplift(model, inputs, baselines, targets, dataset.X.columns)
        leaf_values = feature_deeplift_values
    else:
        for node in tqdm(model.ete_tree.get_leaves(), total=len(model.ete_tree.get_leaves())):
            if args.leaf_attr == "gini_gain":
                gini_gain = compute_gini_gain(dataset, node) 
                leaf_values[node.name] = gini_gain
            elif args.leaf_attr == "pb_correlation":
                pb_correlation = compute_pointbiserialr(dataset, node)
                leaf_values[node.name] = abs(pb_correlation)
            elif args.leaf_attr == "auroc":
                auroc = compute_auroc(dataset, node)
                leaf_values[node.name] = auroc

    if args.layer_deeplift:
        # Create file path for saving attributions
        attribution_file = f"{attributions_dir}/{os.path.basename(model_path)}_{args.num_samples}"
        if args.top_k_nodes is not None:
            attribution_file += f"_{args.top_k_nodes}"
        if args.min_depth is not None:
            attribution_file += f"_{args.min_depth}"
        if args.max_depth is not None:
            attribution_file += f"_{args.max_depth}"
        attribution_file += ".json"

        print(f"Loading attributions from {attribution_file}...")

        # Initialize variables
        layer_deeplift_values = []
        depth = 0
        nodes_to_process = [model.ete_tree.get_tree_root()]

        # Load attributions if they exist, otherwise initialize an empty list
        if os.path.exists(attribution_file):
            with open(attribution_file, 'r') as f:
                layer_deeplift_values = json.load(f)
            if layer_deeplift_values:
                # Resume from the last saved depth
                depth = len(layer_deeplift_values)
                if depth < args.min_depth:
                    nodes_to_process = [model.ete_tree.get_tree_root()]
                else:
                    # Extract the names of the nodes from the last saved depth
                    last_saved_nodes = list(layer_deeplift_values[-1].keys())
                    nodes_to_process = [child for last_saved_node in last_saved_nodes for child in name2node[last_saved_node].get_children()]

        inputs, baselines, targets = prepare_data_for_deeplift(model, dataset, args.num_samples)

        # Checkpoint function for saving progress
        def save_checkpoint():
            with open(attribution_file, 'w') as f:
                json.dump(layer_deeplift_values, f)
            print(f"Checkpoint saved at depth {depth}.")

        # Process nodes
        while nodes_to_process and (not args.max_depth or depth <= args.max_depth):
            current_depth_nodes = nodes_to_process
            depth_results = {}

            # Skip processing for depths less than min_depth
            if depth >= args.min_depth:
                print(f"Processing {len(current_depth_nodes)} nodes at depth {depth}...")
                for node in tqdm(current_depth_nodes, total=len(current_depth_nodes)):
                    attribution = compute_layer_deeplift(model, node, inputs, baselines, targets)
                    depth_results[node.name] = attribution

                # Select top k nodes's children to process
                if args.top_k_nodes:
                    selected_nodes = sorted(depth_results, key=depth_results.get, reverse=True)[:args.top_k_nodes]
                else:
                    selected_nodes = depth_results.keys()
                nodes_to_process = [child for selected_node in selected_nodes for child in name2node[selected_node].get_children()]

                if depth >= args.min_depth:
                    layer_deeplift_values.append(depth_results)
                    if args.save_attributions:
                        save_checkpoint()  # Save progress at each depth

            else:
                # Directly update nodes_to_process for depths less than min_depth
                nodes_to_process = [child for node in nodes_to_process for child in node.get_children()]

            depth += 1

        # Final save after completing all depths
        if args.save_attributions:
            save_checkpoint()
            print("Processing complete. All results saved.")

    return leaf_values, layer_deeplift_values

def node_value(dataset, node):
    if node.is_leaf():
        return dataset.X[node.name]

    # For non-leaf nodes, combine represented features
    represented_features = [child.name for child in node.get_leaves()]
    combined_value = np.log(np.exp(dataset.X[represented_features]).sum(axis=1))
    return combined_value

def compute_gini_gain(dataset, node):
    dt = DecisionTreeClassifier(criterion='gini', max_depth=1)
    node_values = node_value(dataset, node)
    dt.fit(node_values.values.reshape(-1, 1), dataset.y)
    impurity_before_split = dt.tree_.impurity[0]
    impurity_left_child = dt.tree_.impurity[1]
    impurity_right_child = dt.tree_.impurity[2]
    n_node_samples = dt.tree_.n_node_samples
    
    gini_before = impurity_before_split
    gini_after = (n_node_samples[1]/n_node_samples[0])*impurity_left_child + (n_node_samples[2]/n_node_samples[0])*impurity_right_child
    gini_gain = gini_before - gini_after
    return gini_gain

def compute_pointbiserialr(dataset, node):
    node_values = node_value(dataset, node)
    pb_correlation, _ = pointbiserialr(dataset.y, node_values)
    return pb_correlation

def compute_auroc(dataset, node):
    node_values = node_value(dataset, node)
    auroc = roc_auc_score(dataset.y, node_values)
    auroc = 1 - auroc if auroc < 0.5 else auroc
    return auroc


def compute_deeplift(model, inputs, baselines, targets, feature_names):
    deeplift = DeepLift(model)

    # Compute attributions using DeepLift with the baselines and targets
    attributions = deeplift.attribute(inputs, baselines=baselines, target=targets).detach().numpy()
    attributions = np.abs(attributions)

    # Initialize a dictionary to hold the attributions for each feature
    feature_attributions = {}

    # Assign attributions to corresponding features
    for idx, feature_name in enumerate(feature_names):
        feature_attributions[feature_name] = np.mean(attributions[:, idx]).astype(float)

    return feature_attributions


def compute_layer_deeplift(model, node, inputs, baselines, targets):
    model_node = model.nodes[node.name]
    layer_deeplift = LayerDeepLift(model, model_node, multiply_by_inputs=False)

    # Compute attributions using DeepLift with the baselines and targets
    attributions = layer_deeplift.attribute(inputs, baselines=baselines, target=targets).detach().numpy()
    attributions = np.abs(attributions)
    attribution = np.sum(attributions, axis=1).mean().astype(float)

    return attribution

def prepare_data_for_deeplift(model, dataset, num_samples):
    
    selected_indices = np.random.choice(len(dataset), num_samples, replace=False)
    inputs = dataset.X.values[selected_indices]
    targets = dataset.y.values[selected_indices]

    selected_indices = np.random.choice(len(dataset), num_samples, replace=False)
    inputs_list = dataset.X.values[selected_indices]
    baselines_list = np.mean(dataset.X.values, axis=0)
    baselines_list = np.tile(baselines_list, (num_samples, 1))
    targets_list = dataset.y.values[selected_indices]

    inputs = torch.tensor(np.vstack(inputs_list), dtype=torch.float)
    baselines = torch.tensor(np.vstack(baselines_list), dtype=torch.float)
    targets = torch.tensor(targets_list, dtype=torch.long)

    return inputs, baselines, targets


def visualize_node_values(ete_tree, leaf_values, deeplift_values, attributions_dir):

    name2node = {node.name: node for node in ete_tree.traverse('levelorder')}

    # Process node values
    leaf_values, leaf_colors = process_node_values(leaf_values)
    deeplift_values, deeplift_colors = process_node_values(deeplift_values)

    # Print top 6 nodes with the highest DeepLIFT attributions and their top leaf nodes
    sorted_deeplift_values = sorted(deeplift_values.items(), key=lambda x: x[1], reverse=True)
    sorted_leaf_values = sorted(leaf_values.items(), key=lambda x: x[1], reverse=True)
    print("Top 6 internal nodes with the highest DeepLIFT attributions and their top leaf nodes:")
    for node_name, value in sorted_deeplift_values[:6]:
        print(f"{node_name}: {value}")
        for leaf_name, leaf_value in sorted_leaf_values:
            if ete_tree.get_common_ancestor(node_name, leaf_name).name == node_name:
                leaf_parent = name2node[leaf_name].up.name
                print(f"\t{leaf_name}: {leaf_value}")
                print(f"\t{leaf_parent}")
                break

    # Set tree style
    ts = TreeStyle()
    ts.mode = "c"  # circular mode
    ts.show_leaf_name = False
    ts.show_scale = False

    # Set node colors
    for node in ete_tree.traverse():
        node.img_style["fgcolor"] = "lightgrey"
        node.img_style["hz_line_color"] = "lightgrey"
        node.img_style["vt_line_color"] = "lightgrey"
        node.img_style["hz_line_type"] = 0
        node.img_style["vt_line_type"] = 0

    # Color descendants of the selected nodes
    color_descendants(name2node, deeplift_colors)

    # Add rectfaces to leaves
    add_rectfaces(ete_tree, leaf_colors)

    # ete_tree.show(tree_style=ts)

    ete_tree.render(f"{attributions_dir}/fig5_YlGn.png",
                    tree_style=ts, h=2400, units="px", dpi=1200)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(plt.imread(f"{attributions_dir}/fig5_YlGn.png"))

    # save_colorbar(cm.YlGn, 0, 1, "../results/attributions/fig5_YlGn_colorbar.pdf")

def process_node_values(node_values):
    sorted_node_values = sorted(node_values.items(), key=lambda x: x[1], reverse=True)
    
    min_value = min(sorted_node_values, key=lambda x: x[1])[1]
    max_value = max(sorted_node_values, key=lambda x: x[1])[1]

    node_values = {node_name: (value - min_value) / (max_value - min_value) 
                   for node_name, value in sorted_node_values}

    colormap = cm.YlGn
    node_colors = {node_name: to_hex(colormap((value - min_value) / (max_value - min_value)))
                   for node_name, value in sorted_node_values}

    return node_values, node_colors

def color_descendants(name2node, deeplift_colors):
    for node_name, color in deeplift_colors.items():
        node = name2node[node_name]
        for descendant in [node] + node.get_descendants():
            descendant.img_style["fgcolor"] = color
            descendant.img_style["hz_line_color"] = color
            descendant.img_style["vt_line_color"] = color
            descendant.img_style["hz_line_type"] = 0
            descendant.img_style["vt_line_type"] = 0

def add_rectfaces(ete_tree, leaf_colors):
    for leaf in ete_tree.get_leaves():
        rectface = RectFace(width=100, height=5, fgcolor="white", bgcolor="white")
        leaf.add_face(rectface, column=0, position="aligned")
        leaf.add_face(rectface, column=1, position="aligned")
        color = leaf_colors[leaf.name]
        rectface = RectFace(width=200, height=5, fgcolor=color, bgcolor=color)
        leaf.add_face(rectface, column=2, position="aligned")

def save_colorbar(cmap, min_val, max_val, filename):
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cb = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cb.set_label('Attributions')

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def save_to_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load model and data
    model_dir = f"../results/{args.dataset}/{args.target}/models"
    model_path = os.path.join(model_dir, os.listdir(model_dir)[0])
    print(f"Loading model from {model_path}")
    model, _, hyperparams = load_model(model_path)
    
    data, metadata = load_data(f'../data/{args.dataset}', args.dataset)
    _, dataset = preprocess_data(data, metadata, args.dataset, args.target, "WoL2", hyperparams["taxonomy"], 
                                 hyperparams["prob_label"], hyperparams["prune_by_dataset"])
    le = dataset.le
    dataset = dataset.preprocess(hyperparams["preprocess"])
    dataset.le = le

    # Create directory for saving attributions
    attributions_dir = set_attributions_dir(args)

    # Compute attributions
    leaf_values, layer_deeplift_values = compute_attributions(args, model_path, model, dataset, attributions_dir)
   

    if args.visualize:
        visualize_node_values(model.ete_tree, leaf_values, layer_deeplift_values[args.visualize_depth], attributions_dir)

