import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from captum.module import BinaryConcreteStochasticGates, GaussianStochasticGates

class MIOSTONELayer(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 gate_type, 
                 gate_param,
                 connections,
                 prune_mode):
        super(MIOSTONELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_type = gate_type
        self.gate_param = gate_param
        self.connections = connections
        self.prune_mode = prune_mode
        self.x_linear = None
        self.l0_reg = None

        # Initialize the layer
        self._init_layer()

    def _init_layer(self):
        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.LeakyReLU()
        )
        # Linear layer
        self.linear = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
        )

        # Gate layer
        if self.gate_type == "concrete":
            self.gate_mask = self._generate_gate_mask()
            self.gate_layer = BinaryConcreteStochasticGates(n_gates=len(self.connections),
                                                           mask=self.gate_mask,
                                                           temperature=self.gate_param)
        elif self.gate_type == "gaussian":
            self.gate_mask = self._generate_gate_mask()
            self.gate_layer = GaussianStochasticGates(n_gates=len(self.connections),
                                                        mask=self.gate_mask,
                                                        std=self.gate_param)
            
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
        # If the prune mode is random, generate random connections
        if self.prune_mode == "random":
            self._generate_random_connections()
        # Define a custom prune method for each layer
        prune.custom_from_mask(self.mlp[0], name='weight', mask=self._generate_pruning_mask())
        prune.custom_from_mask(self.linear[0], name='weight', mask=self._generate_pruning_mask())
        # Remove the original weight parameter
        prune.remove(self.mlp[0], 'weight')
        prune.remove(self.linear[0], 'weight')

    def _generate_random_connections(self):
        connections = {}
        all_input_indices = [mapping[0] for mapping in self.connections.values()]
        for ete_node, (_, output_indices) in self.connections.items():
            idx = random.randint(0, len(all_input_indices) - 1)
            input_indices = all_input_indices[idx]
            connections[ete_node] = (input_indices, output_indices)
            all_input_indices = all_input_indices[:idx] + all_input_indices[idx + 1:]

        self.connections = connections

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
        x_mlp = self.mlp(x)

        # Apply the linear layer
        self.x_linear = self.linear(x_linear)
        
        # Apply the linear layer with the gate values
        if self.gate_type == "deterministic":
            gate_values = self.gate_param
            self.l0_reg = torch.tensor(0.0).to(x.device)
        else:
            input_size = x_mlp.size()
            batch_size = input_size[0]

            gate_values = self.gate_layer._sample_gate_values(batch_size)

            # hard-sigmoid rectification z=min(1,max(0,_z))
            gate_values = torch.clamp(gate_values, min=0, max=1)

            # use expand_as not expand/broadcast_to which do not work with torch.fx
            input_mask = self.gate_layer.mask.expand_as(x_mlp)

            # flatten all dim except batch to gather from gate values
            flattened_mask = input_mask.reshape(batch_size, -1)
            gate_values = torch.gather(gate_values, 1, flattened_mask)

            # reshape gates(batch_size, n_elements) into input_size for point-wise mul
            gate_values = gate_values.reshape(input_size)

            prob_density = self.gate_layer._get_gate_active_probs()
            if self.gate_layer.reg_reduction == "sum":
                l0_reg = prob_density.sum()
            elif self.gate_layer.reg_reduction == "mean":
                l0_reg = prob_density.mean()
            else:
                l0_reg = prob_density

            l0_reg *= self.gate_layer.reg_weight
            self.l0_reg = l0_reg

        # Apply the gate values
        x_mlp_gated = gate_values * x_mlp
        x_linear_gated = (1 - gate_values) * self.x_linear

        x_gated = x_mlp_gated + x_linear_gated

        return x_gated


class MIOSTONEModel(nn.Module):
    def __init__(self, 
                 tree,
                 out_features,
                 node_min_dim,
                 node_dim_func,
                 node_dim_func_param, 
                 node_gate_type,
                 node_gate_param,
                 prune_mode):
        super(MIOSTONEModel, self).__init__()
        self.out_features = out_features
        self.node_min_dim = node_min_dim
        self.node_dim_func = node_dim_func
        self.node_dim_func_param = node_dim_func_param
        self.node_gate_type = node_gate_type
        self.node_gate_param = node_gate_param
        self.prune_mode = prune_mode
        self.hidden_layers = None
        self.output_layer = None
        self.total_l0_reg = None

        # Initialize the architecture based on the tree
        connections, layer_dims = self._init_architecture(tree)

        # Build the model based on the architecture
        self._build_model(connections, layer_dims)

    def _init_architecture(self, tree):
        # Define the node dimension function
        def dim_func(x, node_dim_func, node_dim_func_param, depth):
            if node_dim_func == "linear":
                coeff = node_dim_func_param ** (tree.max_depth - depth)
                return int(coeff * x)
            elif node_dim_func == "const":
                return int(node_dim_func_param)

        # Initialize dictionary for connections and layer dimensions
        layer_connections = [{} for _ in range(tree.max_depth + 1)]
        layer_dims = [None for _ in range(tree.max_depth + 1)]

        curr_index = 0
        curr_depth = tree.max_depth
        prev_layer_out_features = 0

        for ete_node in reversed(list(tree.ete_tree.traverse("levelorder"))):
            node_depth = tree.depths[ete_node.name]
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
                                  connections,
                                  prune_mode=self.prune_mode)
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
        for layer in reversed(self.hidden_layers):
            # Apply the layer
            x = layer(x, x_linear)

            # Update the linear layer input
            x_linear = layer.x_linear
            layer.x_linear = None

            # Update the total l0 regularization
            self.total_l0_reg += layer.l0_reg
            layer.l0_reg = None

        # Apply the output layer
        x = self.output_layer(x)

        return x
    
    def get_total_l0_reg(self):
       return self.total_l0_reg