import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.fc2 = nn.Linear(in_features // 2, in_features // 4)
        self.output_layer = nn.Linear(in_features // 4, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output_layer(x)
        return x
    
    def get_total_l0_reg(self):
        return torch.tensor(0.0).to(self.fc1.weight.device)

class TaxoNN(nn.Module):
    def __init__(self, tree, out_features, data):
        super(TaxoNN, self).__init__()
        self.out_features = out_features

        # Initialize the stratified indices 
        self._init_stratification(tree, data)
        
        # Build the model based on the stratified indices
        self._build_model(tree)
    
    def _init_stratification(self, tree, data):
        stratified_indices = {ete_node: [] for ete_node in tree.ete_tree.traverse("levelorder") if tree.depths[ete_node.name] == 2}
        descendants = {ete_node: set(ete_node.descendants()) for ete_node in stratified_indices.keys()}

        for i, leaf_node in enumerate(tree.ete_tree.leaves()):
            for ete_node in stratified_indices.keys():
                if leaf_node in descendants[ete_node]:
                    stratified_indices[ete_node].append(i)
                    break

        self.stratified_indices = stratified_indices
        self._order_stratified_indices(data)
    
    def _order_stratified_indices(self, data):
        for ete_node, indices in self.stratified_indices.items():
            # Get the data for the current cluster
            cluster = data.X[:, indices]

            # Skip if there is only one feature
            if cluster.shape[1] == 1:
                continue

            # Calculate Spearman correlation matrix
            corr_matrix, _ = spearmanr(cluster)

            # Sum of correlations for each feature
            corr_sum = np.sum(corr_matrix, axis=0)

            # Sort indices based on correlation sum
            sorted_indices = np.argsort(corr_sum)

            # Update the indices in the stratified_indices dictionary
            self.stratified_indices[ete_node] = [indices[i] for i in sorted_indices]

    def _build_model(self, tree):
        self.cnn_layers = nn.ModuleDict()
        for ete_node in self.stratified_indices.keys():
            self.cnn_layers[ete_node.name] = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, padding=1),
                nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, padding=1),
                nn.Flatten()
            )
        output_layer_in_features = self._compute_output_layer_in_features(tree)
        self.output_layer = nn.Sequential(
            nn.Linear(output_layer_in_features, 100),
            nn.ReLU(), 
            nn.Linear(100, self.out_features))
        
    def _compute_output_layer_in_features(self, tree):
        dummy_input = torch.zeros((1, len(list(tree.ete_tree.leaves()))))
        output_in_features = 0
        for ete_node, indices in self.stratified_indices.items():
            data = dummy_input[:, indices]
            data = data.unsqueeze(1)
            output_in_features += self.cnn_layers[ete_node.name](data).shape[1]
        return output_in_features

        
    def forward(self, x):
        # Iterate over the CNNs and apply them to the corresponding data
        outputs = []
        for ete_node, indices in self.stratified_indices.items():
            data = x[:, indices]
            data = data.unsqueeze(1)
            data = self.cnn_layers[ete_node.name](data)
            outputs.append(data)

        # Concatenate the outputs from the CNNs
        outputs = torch.cat(outputs, dim=1)

        # Apply the output layer
        x = self.output_layer(outputs)

        return x
    
    def get_total_l0_reg(self):
        return torch.tensor(0.0).to(self.output_layer[0].weight.device)

class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

class PopPhyCNN(nn.Module):
    def __init__(self, 
                 tree,
                 out_features, 
                 num_kernel, 
                 kernel_height, 
                 kernel_width, 
                 num_fc_nodes, 
                 num_cnn_layers, 
                 num_fc_layers, 
                 dropout):
        super(PopPhyCNN, self).__init__()
        self.out_features = out_features
        self.num_kernel = num_kernel
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.num_fc_nodes = num_fc_nodes
        self.num_cnn_layers = num_cnn_layers
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout

        self._build_model(tree)

    def _build_model(self, tree):
        self.gaussian_noise = GaussianNoise(0.01)
        self.cnn_layers = self._create_conv_layers()
        self.fc_layers = self._create_fc_layers(tree)
        self.output_layer = nn.Linear(self.num_fc_nodes, self.out_features)

    def _create_conv_layers(self):
        layers = []
        for i in range(self.num_cnn_layers):
            in_channels = 1 if i == 0 else self.num_kernel
            layers.append(nn.Conv2d(in_channels, self.num_kernel, (self.kernel_height, self.kernel_width)))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(nn.Flatten())
        layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)

    def _create_fc_layers(self, tree):
        layers = []
        for i in range(self.num_fc_layers):
            fc_in_features = self._compute_fc_layer_in_features(tree) if i == 0 else self.num_fc_nodes
            layers.append(nn.Linear(fc_in_features, self.num_fc_nodes))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)
    
    def _compute_fc_layer_in_features(self, tree):
        num_rows = tree.max_depth + 1
        num_cols = len(list(tree.ete_tree.leaves()))
        dummy_input = torch.zeros((1, num_rows, num_cols))
        dummy_input = dummy_input.unsqueeze(1)
        return self.cnn_layers(dummy_input).shape[1]

    def forward(self, x):
        x = self.gaussian_noise(x.unsqueeze(1))
        x = self.cnn_layers(x)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x

    def get_total_l0_reg(self):
        return torch.tensor(0.0).to(self.output_layer.weight.device)
