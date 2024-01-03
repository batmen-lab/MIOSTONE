import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from captum.module import BinaryConcreteStochasticGates
from scipy.stats import spearmanr


class DeterministicGate(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values
    
    def forward(self, x):
        return x * self.values, torch.tensor(0.0).to(self.values.device)
    
    def get_gate_values(self):
        return self.values

class MIOSTONEModel(nn.Module):
    def __init__(self, 
                 tree,
                 out_features,
                 node_min_dim,
                 node_dim_func,
                 node_dim_func_param, 
                 node_gate_type,
                 node_gate_param):
        super().__init__()
        self.tree = tree
        self.out_features = out_features
        self.node_min_dim = node_min_dim
        self.node_dim_func = node_dim_func
        self.node_dim_func_param = node_dim_func_param
        self.node_gate_type = node_gate_type
        self.node_gate_param = node_gate_param

        # Initialize the architecture based on the tree
        self._init_architecture_from_tree()

        # Build the model based on the architecture
        self._build_model()

    def _init_architecture_from_tree(self):
        # Define the node dimension function
        def dim_func(x, node_dim_func, node_dim_func_param, depth):
            if node_dim_func == "linear":
                coeff = node_dim_func_param ** (self.tree.max_depth - depth)
                return int(coeff * x)
            elif node_dim_func == "const":
                return int(node_dim_func_param)

        # Initialize dictionary for connections and layer dimensions
        self.connections = [{} for _ in range(self.tree.max_depth + 1)]
        self.layer_dims = [None for _ in range(self.tree.max_depth + 1)]

        curr_index = 0
        curr_depth = self.tree.max_depth
        prev_layer_out_features = 0

        for ete_node in reversed(list(self.tree.ete_tree.traverse("levelorder"))):
            node_depth = self.tree.depths[ete_node.name]
            if node_depth != curr_depth:
                self.layer_dims[curr_depth] = (curr_index, prev_layer_out_features)
                curr_depth = node_depth
                prev_layer_out_features = curr_index
                curr_index = 0

            if ete_node.is_leaf:
                self.connections[curr_depth][ete_node.name] = ([], [curr_index])
                curr_index += 1
                continue

            children = ete_node.get_children()

            # Calculate input indices
            input_indices = []
            for child in children:
                child_output_indices = self.connections[node_depth + 1][child.name][1]
                input_indices.extend(child_output_indices)

            # Calculate output dimensions and indices
            node_out_features = max(self.node_min_dim, 
                                    dim_func(self.node_min_dim * len(list(ete_node.leaves())),
                                            self.node_dim_func, 
                                            self.node_dim_func_param, 
                                            node_depth))
            output_indices = list(range(curr_index, curr_index + node_out_features))
            curr_index += node_out_features

            # Store in connections
            self.connections[curr_depth][ete_node.name] = (input_indices, output_indices)

        # Append the dimension of the last layer
        self.layer_dims[0] = (curr_index, prev_layer_out_features)

        # Remove the layer dimension of the leaf nodes
        self.layer_dims = self.layer_dims[:-1]

    def _build_model(self):
        self.mlp_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        self.gate_masks = []

        # Initialize the internal layers
        for depth, (out_features, in_features) in enumerate(self.layer_dims):
            # Initialize the mlp layer
            mlp_layer = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
            )
            self.mlp_layers.append(mlp_layer)

            # Initialize the linear layer
            linear_layer = nn.Linear(in_features, out_features)
            self.linear_layers.append(linear_layer)

            # Initialize the gate layer
            gate_mask = self._generate_gate_mask(depth)
            if self.node_gate_type == "deterministic":
                gate_layer = DeterministicGate(self.node_gate_param)
            elif self.node_gate_type == "concrete":
                gate_layer = BinaryConcreteStochasticGates(n_gates=len(self.connections[depth]), 
                                                           mask=gate_mask,
                                                           temperature=self.node_gate_param)
            self.gate_masks.append(gate_mask)
            self.gate_layers.append(gate_layer)

        # Initialize the output layer
        output_layer_in_features = self.layer_dims[0][0] 
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(output_layer_in_features),
            nn.Linear(output_layer_in_features, self.out_features)
        )

        # Prune the network based on the connections
        self._apply_pruning()

    def _generate_gate_mask(self, depth):
        mask = torch.zeros(self.layer_dims[depth][0], dtype=torch.int64)
        value = 0
        for _, output_indices in self.connections[depth].values():
            for output_index in output_indices:
                mask[output_index] = value
            value += 1

        return mask

    def _apply_pruning(self):
        for depth, (mlp_layer, linear_layer) in enumerate(zip(self.mlp_layers, self.linear_layers)):
            # Define a custom prune method for each layer
            prune.custom_from_mask(mlp_layer[0], name='weight', mask=self._generate_pruning_mask(depth))
            prune.custom_from_mask(linear_layer, name='weight', mask=self._generate_pruning_mask(depth))
            # Remove the original weight parameter
            prune.remove(mlp_layer[0], 'weight')
            prune.remove(linear_layer, 'weight')

    def _generate_pruning_mask(self, depth):
        # Start with a mask of all zeros (all connections pruned)
        mask = torch.zeros(self.layer_dims[depth])

        # Iterate over the connections at the current depth and set the corresponding elements in the mask to 1
        for input_indices, output_indices in self.connections[depth].values():
            for input_index in input_indices:
                for output_index in output_indices:
                    mask[output_index, input_index] = 1

        return mask
    
    def forward(self, x):
        # Initialize the total l0 regularization
        self.total_l0_reg = torch.tensor(0.0).to(x.device)

        # Initialize the linear layer input
        x_linear = x

        # Iterate over the layers
        for depth in reversed(range(self.tree.max_depth)):
            # Get the layers
            mlp_layer = self.mlp_layers[depth]
            linear_layer = self.linear_layers[depth]
            gate_layer = self.gate_layers[depth]
            gate_mask = self.gate_masks[depth]

            # Apply the mlp layer
            x = mlp_layer(x)

            # Apply the linear layer
            x_linear = linear_layer(x_linear)

            # Apply the gate layer
            x, l0_reg = gate_layer(x)
            self.total_l0_reg += l0_reg

            # Apply the linear layer with the gate values
            gate_values = gate_layer.get_gate_values()
            gate_values = self._reshape_gate_values(gate_values, gate_mask)
            x = x + (1 - gate_values) * x_linear

        # Apply the output layer
        x = self.output_layer(x)

        return x
    
    def _reshape_gate_values(self, gate_values, gate_mask):
        new_gate_values = torch.zeros(gate_mask.shape).to(gate_values.device)
        for i, value in enumerate(gate_values):
            new_gate_values[gate_mask == i] = value
        return new_gate_values
    
    def get_total_l0_reg(self):
       return self.total_l0_reg

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.fc2 = nn.Linear(in_features // 2, in_features // 4)
        self.fc3 = nn.Linear(in_features // 4, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_total_l0_reg(self):
        return torch.tensor(0.0).to(self.fc1.weight.device)

class TaxoNN(nn.Module):
    def __init__(self, tree, out_features, dataset):
        super().__init__()
        self.tree = tree
        self.out_features = out_features

        # Initialize the stratified indices 
        self._init_stratification(dataset)
        
        # Build the model based on the stratified indices
        self._build_model()
    
    def _init_stratification(self, dataset):
        stratified_indices = {ete_node: [] for ete_node in self.tree.ete_tree.traverse("levelorder") if self.tree.depths[ete_node.name] == 2}
        descendants = {ete_node: set(ete_node.descendants()) for ete_node in stratified_indices.keys()}

        for i, leaf_node in enumerate(self.tree.ete_tree.leaves()):
            for ete_node in stratified_indices.keys():
                if leaf_node in descendants[ete_node]:
                    stratified_indices[ete_node].append(i)
                    break

        self.stratified_indices = stratified_indices
        self._order_stratified_indices(dataset)
    
    def _order_stratified_indices(self, dataset):
        for ete_node, indices in self.stratified_indices.items():
            # Extract the data for the current group
            data = dataset.X[:, indices]

            # Skip if there is only one feature
            if data.shape[1] == 1:
                continue

            # Calculate Spearman correlation matrix
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try :
                    corr_matrix, _ = spearmanr(data)
                except Warning:
                    print(f"Data: {data}")

            # Sum of correlations for each feature
            corr_sum = np.sum(corr_matrix, axis=0)

            # Sort indices based on correlation sum
            sorted_indices = np.argsort(corr_sum)

            # Update the indices in the stratified_indices dictionary
            self.stratified_indices[ete_node] = [indices[i] for i in sorted_indices]

    def _build_model(self):
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
        output_layer_in_features = self._compute_output_layer_in_features()
        self.output_layer = nn.Sequential(
            nn.Linear(output_layer_in_features, 100),
            nn.ReLU(), 
            nn.Linear(100, self.out_features))
        
    def _compute_output_layer_in_features(self):
        dummy_input = torch.zeros((1, len(list(self.tree.ete_tree.leaves()))))
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
        self.tree = tree
        self.out_features = out_features
        self.num_kernel = num_kernel
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.num_fc_nodes = num_fc_nodes
        self.num_cnn_layers = num_cnn_layers
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout

        self._build_model()

    def _build_model(self):
        self.cnn_layers = self._create_conv_layers()
        self.fc_layers = self._create_fc_layers()
        self.output_layer = nn.Linear(self.num_fc_nodes, self.out_features)

    def _create_conv_layers(self):
        layers = []
        for i in range(self.num_cnn_layers):
            in_channels = 1 if i == 0 else self.num_kernel
            layers.append(nn.Conv2d(in_channels, self.num_kernel, (self.kernel_height, self.kernel_width), padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(nn.Flatten())
        layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)

    def _create_fc_layers(self):
        layers = []
        for i in range(self.num_fc_layers):
            fc_in_features = self._compute_fc_layer_in_features() if i == 0 else self.num_fc_nodes
            layers.append(nn.Linear(fc_in_features, self.num_fc_nodes))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)
    
    def _compute_fc_layer_in_features(self):
        dummy_input = torch.zeros((1, self.tree.max_depth + 1, len(list(self.tree.ete_tree.leaves()))))
        dummy_input = dummy_input.unsqueeze(1)
        return self.cnn_layers(dummy_input).shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn_layers(x)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x

    def get_total_l0_reg(self):
        return torch.tensor(0.0).to(self.output_layer.weight.device)
