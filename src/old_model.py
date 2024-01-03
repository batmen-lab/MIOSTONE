import torch
import torch.nn as nn
from captum.module import (BinaryConcreteStochasticGates,
                           GaussianStochasticGates)


class DeterministicGate(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values
    
    def forward(self, x):
        return x * self.values, torch.tensor(0.0).to(self.values.device)
    
    def get_gate_values(self):
        return self.values
    
class TreeNode(nn.Module):
    def __init__(self, name, in_features, out_features, gate_type, gate_param):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features

        if gate_type == 'gaussian':
            self.gate = GaussianStochasticGates(n_gates=1, mask=torch.tensor([0]), temperature=gate_param)
        elif gate_type == 'concrete':
            self.gate = BinaryConcreteStochasticGates(n_gates=1, mask=torch.tensor([0]), temperature=gate_param)
        elif gate_type == 'deterministic':
            self.gate = DeterministicGate(torch.tensor(gate_param))
        else:
            raise ValueError(f"Invalid gate type: {gate_type}")
        
        self.mlp_embedding_layer = nn.Sequential(nn.Linear(self.in_features, self.out_features), nn.ReLU())
        self.linear_embedding_layer = nn.Linear(self.in_features, self.out_features)

    def forward(self, children_embeddings_mlp, children_embeddings_linear):        
        # Compute the node embedding 
        node_embedding_mlp = self.mlp_embedding_layer(children_embeddings_mlp)
        node_embedding_linear = self.linear_embedding_layer(children_embeddings_linear)
        self.node_embedding_linear = node_embedding_linear
       
        # Compute the gate values
        gated_node_embedding_mlp, l0_reg = self.gate(node_embedding_mlp)
        self.l0_reg = l0_reg
        
        # Compute the gated node embedding
        gated_node_embedding = gated_node_embedding_mlp + (1 - self.gate.get_gate_values()) * node_embedding_linear

        return gated_node_embedding
    
    def get_node_embedding_linear(self):
        return self.node_embedding_linear
    
    def get_l0_reg(self):
        return self.l0_reg

class TreeModel(nn.Module):
    def __init__(self, 
                 tree,
                 out_features,
                 node_min_features=1, 
                 node_out_features_func='linear',
                 node_out_features_func_param=0.95, 
                 node_gate_type='concrete',
                 node_gate_param=0.1):
        super().__init__()
        self.tree = tree
        self.out_features = out_features
        self.node_min_features = node_min_features
        self.node_out_features_func = node_out_features_func
        self.node_out_features_func_param = node_out_features_func_param
        self.node_gate_type = node_gate_type
        self.node_gate_param = node_gate_param

        # Build the network based on the ETE Tree
        self._build_network()

    def _build_network(self):
        self.nodes = nn.ModuleDict()

        # Define the node dimension function
        def dim_func(x, node_out_features_func, node_out_features_func_param, depth):
            if node_out_features_func == "linear":
                coeff = node_out_features_func_param ** (self.tree.max_depth - depth)
                return int(coeff * x)
            elif node_out_features_func == "const":
                return int(node_out_features_func_param)
        
        # Iterate over nodes in ETE Tree and build TreeNode
        self.out_features_dict = {}
        for ete_node in reversed(list(self.tree.ete_tree.traverse("levelorder"))):
            # Skip if the node is a leaf
            if ete_node.is_leaf:
                self.out_features_dict[ete_node.name] = 1
                continue
            
            # Skip if the node is a unary node
            children = ete_node.get_children()
            if len(children) == 1 and not children[0].is_leaf:
                self.out_features_dict[ete_node.name] = self.out_features_dict[children[0].name]
                continue

            # Set the input dimensions
            children_out_features = []
            for child in ete_node.get_children():
                children_out_features.append(self.out_features_dict[child.name])
            node_in_features = sum(children_out_features)

            # Set the output dimensions
            node_out_features = max(self.node_min_features, 
                                    dim_func(self.node_min_features * len(list(ete_node.leaves())),
                                                self.node_out_features_func, 
                                                self.node_out_features_func_param, 
                                                self.tree.depths[ete_node.name]))
            self.out_features_dict[ete_node.name] = node_out_features

            # if node_out_features != self.node_min_features:
            #    print(f"{ete_node.name}: {node_in_features} -> {node_out_features}")

            # Build the node
            tree_node = TreeNode(ete_node.name, node_in_features, node_out_features, self.node_gate_type, self.node_gate_param)
            self.nodes[ete_node.name] = tree_node

        # Build batch normalization layers
        self.batch_norm_layers = nn.ModuleDict()
        for depth in range(self.tree.max_depth):
            self.batch_norm_layers[str(depth)] = nn.BatchNorm1d(sum([self.out_features_dict[ete_node.name] 
                                                                for ete_node in self.tree.ete_tree.traverse("levelorder") 
                                                                if self.tree.depths[ete_node.name] == depth]))

        # Build output layer
        in_features = self.nodes[self.tree.ete_tree.name].out_features
        self.output_layer = nn.Linear(in_features, self.out_features)

    def forward(self, x):
        # Initialize a dictionary to store the outputs at each node
        self.outputs = [{} for _ in range(self.tree.max_depth + 1)]
        self.total_l0_reg = torch.tensor(0.0).to(x.device)

        # Perform a forward pass on the leaves
        input_split = torch.split(x, split_size_or_sections=1, dim=1)
        for leaf_node, input in zip(self.tree.ete_tree.leaves(), input_split):
            embedding_mlp, embedding_linear = input, input
            self.outputs[self.tree.depths[leaf_node.name]][leaf_node.name] = (embedding_mlp, embedding_linear)
            
        # Perform a forward pass on the internal nodes in "topological order"
        curr_depth = self.tree.max_depth - 1
        for ete_node in reversed(list(self.tree.ete_tree.traverse("levelorder"))):
            if not ete_node.is_leaf:
                children = ete_node.get_children()
                depth = self.tree.depths[ete_node.name]
                if len(children) == 1 and not children[0].is_leaf:
                    self.outputs[depth][ete_node.name] = self.outputs[depth + 1][children[0].name]
                else:
                    children_outputs = [self.outputs[depth + 1][child.name] for child in children]
                    children_embeddings_mlp = torch.cat([out[0] for out in children_outputs], dim=1)
                    children_embeddings_linear = torch.cat([out[1] for out in children_outputs], dim=1)
                    embedding_mlp = self.nodes[ete_node.name](children_embeddings_mlp, children_embeddings_linear)
                    embedding_linear = self.nodes[ete_node.name].get_node_embedding_linear()
                    self.outputs[depth][ete_node.name] = (embedding_mlp, embedding_linear)
                    self.total_l0_reg += self.nodes[ete_node.name].get_l0_reg()
                if depth != curr_depth:
                    concatenated_embeddings = torch.cat([out[0] for out in self.outputs[curr_depth].values()], dim=1)
                    concatenated_embeddings = self.batch_norm_layers[str(curr_depth)](concatenated_embeddings)
                    for ete_node in self.outputs[curr_depth].keys():
                        self.outputs[curr_depth][ete_node] = (concatenated_embeddings[:, :self.out_features_dict[ete_node]],
                                                              concatenated_embeddings[:, :self.out_features_dict[ete_node]])
                    curr_depth -= 1

        # Pass the tree out through the final out layer
        logits = self.output_layer(self.outputs[0][self.tree.ete_tree.name][0])

        return logits
    
    def get_total_l0_reg(self):
        return self.total_l0_reg

   