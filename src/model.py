import numpy as np
import torch
import torch.nn as nn
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
    
class MIOSTONELayer(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 gate_type, 
                 gate_param,
                 connections):
        super(MIOSTONELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_type = gate_type
        self.gate_param = gate_param
        self.connections = connections
        self.x_linear = None
        self.l0_reg = None

        # Initialize the layer
        self._init_layer()

    def _init_layer(self):
        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.ReLU(),
        )
        # Linear layer
        self.linear = nn.Linear(self.in_features, self.out_features)

        # Gate layer
        if self.gate_type == "deterministic":
            self.gate_layer = DeterministicGate(self.gate_param)
        elif self.gate_type == "concrete":
            self.gate_mask = self._generate_gate_mask()
            self.gate_layer = BinaryConcreteStochasticGates(n_gates=len(self.connections),
                                                           mask=self.gate_mask,
                                                           temperature=self.gate_param)
            
        # Prune the network based on the connections
        self._apply_pruning()


    def _generate_gate_mask(self):
        mask = torch.zeros(self.out_features, dtype=torch.int64)
        value = 0
        for _, output_indices in self.connections.values():
            for output_index in output_indices:
                mask[output_index] = value
            value += 1

        return mask

    def _apply_pruning(self):
        # Define a custom prune method for each layer
        prune.custom_from_mask(self.mlp[0], name='weight', mask=self._generate_pruning_mask())
        prune.custom_from_mask(self.linear, name='weight', mask=self._generate_pruning_mask())
        # Remove the original weight parameter
        prune.remove(self.mlp[0], 'weight')
        prune.remove(self.linear, 'weight')

    def _generate_pruning_mask(self):
        # Start with a mask of all zeros (all connections pruned)
        mask = torch.zeros((self.out_features, self.in_features), dtype=torch.int64)

        # Iterate over the connections at the current depth and set the corresponding elements in the mask to 1
        for input_indices, output_indices in self.connections.values():
            for input_index in input_indices:
                for output_index in output_indices:
                    mask[output_index, input_index] = 1

        return mask

    def forward(self, x, x_linear):
        # Apply the MLP layer
        x = self.mlp(x)

        # Apply the linear layer
        self.x_linear = self.linear(x_linear)

        # Apply the gate layer
        x, self.l0_reg = self.gate_layer(x)
        
        # Apply the linear layer with the gate values
        gate_values = self.gate_layer.get_gate_values()
        if isinstance(self.gate_layer, BinaryConcreteStochasticGates):
            gate_values = torch.gather(gate_values, 0, self.gate_mask.to(gate_values.device))
        x = x + (1 - gate_values) * self.x_linear

        return x

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
        self.hidden_layers = None
        self.output_layer = None
        self.total_l0_reg = None

        # Initialize the architecture based on the tree
        connections, layer_dims = self._init_architecture_from_tree()

        # Build the model based on the architecture
        self._build_model(connections, layer_dims)

    def _init_architecture_from_tree(self):
        # Define the node dimension function
        def dim_func(x, node_dim_func, node_dim_func_param, depth):
            if node_dim_func == "linear":
                coeff = node_dim_func_param ** (self.tree.max_depth - depth)
                return int(coeff * x)
            elif node_dim_func == "const":
                return int(node_dim_func_param)

        # Initialize dictionary for connections and layer dimensions
        layer_connections = [{} for _ in range(self.tree.max_depth + 1)]
        layer_dims = [None for _ in range(self.tree.max_depth + 1)]

        curr_index = 0
        curr_depth = self.tree.max_depth
        prev_layer_out_features = 0

        for ete_node in reversed(list(self.tree.ete_tree.traverse("levelorder"))):
            node_depth = self.tree.depths[ete_node.name]
            if node_depth != curr_depth:
                layer_dims[curr_depth] = (prev_layer_out_features, curr_index)
                curr_depth = node_depth
                prev_layer_out_features = curr_index
                curr_index = 0

            if ete_node.is_leaf:
                layer_connections[curr_depth][ete_node.name] = ([], [curr_index])
                curr_index += 1
                continue

            children = ete_node.get_children()

            # Calculate input indices
            input_indices = []
            for child in children:
                child_output_indices = layer_connections[node_depth + 1][child.name][1]
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
            layer_connections[curr_depth][ete_node.name] = (input_indices, output_indices)

        # Append the dimension of the last layer
        layer_dims[0] = (prev_layer_out_features, curr_index)

        # Remove the layer dimension of the leaf nodes
        layer_dims = layer_dims[:-1]

        return layer_connections, layer_dims

    def _build_model(self, layer_connections, layer_dims):
        # Initialize the hidden layers
        self.hidden_layers = nn.ModuleList()
        for depth, (in_features, out_features) in enumerate(layer_dims):
            # Get the connections for the current layer
            connections = layer_connections[depth]

            # Initialize the layer
            layer = MIOSTONELayer(in_features, 
                                  out_features, 
                                  self.node_gate_type, 
                                  self.node_gate_param, 
                                  connections)
            self.hidden_layers.append(layer)
            
        # Initialize the output layer
        output_layer_in_features = layer_dims[0][1] 
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(output_layer_in_features),
            nn.Linear(output_layer_in_features, self.out_features)
        )
    
    def forward(self, x):
        # Initialize the total l0 regularization
        self.total_l0_reg = torch.tensor(0.0).to(x.device)

        # Initialize the linear layer input
        x_linear = x

        # Iterate over the layers
        for depth in reversed(range(self.tree.max_depth)):
            # Apply the layer
            x = self.hidden_layers[depth](x, x_linear)

            # Update the linear layer input
            x_linear = self.hidden_layers[depth].x_linear
            self.hidden_layers[depth].x_linear = None

            # Update the total l0 regularization
            self.total_l0_reg += self.hidden_layers[depth].l0_reg
            self.hidden_layers[depth].l0_reg = None

        # Apply the output layer
        x = self.output_layer(x)

        return x
    
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
            corr_matrix, _ = spearmanr(data)

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
        self.gaussian_noise = GaussianNoise(0.01)
        self.cnn_layers = self._create_conv_layers()
        self.fc_layers = self._create_fc_layers()
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
        x = self.gaussian_noise(x.unsqueeze(1))
        x = self.cnn_layers(x)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x

    def get_total_l0_reg(self):
        return torch.tensor(0.0).to(self.output_layer.weight.device)
