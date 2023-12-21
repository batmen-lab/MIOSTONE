import torch
import torch.nn as nn
from captum.module import (BinaryConcreteStochasticGates,
                           GaussianStochasticGates)


class MLPModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.fc2 = nn.Linear(in_features // 2, in_features // 4)
        self.fc3 = nn.Linear(in_features // 4, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
    def get_total_l0_reg(self):
        return torch.tensor(0.0).to(self.fc1.weight.device)

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

        self.mlp_embedding_layer = nn.Sequential(nn.Linear(self.in_features, self.out_features), nn.ReLU())
        self.linear_embedding_layer = nn.Linear(self.in_features, self.out_features)

        if gate_type == 'gaussian':
            self.gate = GaussianStochasticGates(n_gates=1, mask=torch.tensor([0]), temperature=gate_param)
        elif gate_type == 'concrete':
            self.gate = BinaryConcreteStochasticGates(n_gates=1, mask=torch.tensor([0]), temperature=gate_param)
        elif gate_type == 'deterministic':
            self.gate = DeterministicGate(torch.tensor(gate_param))
        else:
            raise ValueError(f"Invalid gate type: {gate_type}")

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
                 node_min_dim=1, 
                 node_dim_func='linear',
                 node_dim_func_param=0.95, 
                 node_gate_type='concrete',
                 node_gate_param=0.1):
        super().__init__()
        self.tree = tree
        self.out_features = out_features
        self.node_min_dim = node_min_dim
        self.node_dim_func = node_dim_func
        self.node_dim_func_param = node_dim_func_param
        self.node_gate_type = node_gate_type
        self.node_gate_param = node_gate_param

        # Build the network based on the ETE Tree
        self._build_network()

    def _build_network(self):
        self.nodes = nn.ModuleDict()
        self.embedding_nodes = [] 

        # Define the node dimension function
        def dim_func(x, node_dim_func, node_dim_func_param, depth):
            if node_dim_func == "linear":
                coeff = node_dim_func_param ** (self.tree.max_depth - depth)
                return int(coeff * x)
            elif node_dim_func == "const":
                return int(node_dim_func_param)
        
        # Iterate over nodes in ETE Tree and build TreeNode
        for ete_node in reversed(list(self.tree.ete_tree.traverse("levelorder"))):
            # Set the input dimensions
            children_out_dims = [self.nodes[child.name].out_features for child in ete_node.get_children()] if not ete_node.is_leaf else [1]
            node_in_dim = sum(children_out_dims)

            # Set the out dimensions
            node_out_dim = max(self.node_min_dim, dim_func(len(list(ete_node.leaves())), self.node_dim_func, self.node_dim_func_param, self.tree.depths[ete_node.name]))

            # Build the node
            tree_node = TreeNode(ete_node.name,node_in_dim, node_out_dim, self.node_gate_type, self.node_gate_param)
            self.nodes[ete_node.name] = tree_node

        # Build out layer
        out_in_dim = self.nodes[self.tree.ete_tree.name].out_features
        self.out_layer = nn.Sequential(nn.BatchNorm1d(out_in_dim), nn.Linear(out_in_dim, self.out_features))

    def forward(self, x):
        # Initialize a dictionary to store the outputs at each node
        self.outputs = {}
        self.total_l0_reg = torch.tensor(0.0).to(x.device)

        # Perform a forward pass on the leaves
        input_split = torch.split(x, split_size_or_sections=1, dim=1)
        for leaf_node, input in zip(self.tree.ete_tree.leaves(), input_split):
            embedding_mlp = self.nodes[leaf_node.name](input, input)
            embedding_linear = self.nodes[leaf_node.name].get_node_embedding_linear()
            self.outputs[leaf_node.name] = (embedding_mlp, embedding_linear)
            self.total_l0_reg += self.nodes[leaf_node.name].get_l0_reg()
            
        # Perform a forward pass on the internal nodes in "topological order"
        for ete_node in reversed(list(self.tree.ete_tree.traverse("levelorder"))):
            if not ete_node.is_leaf:
                children_outputs = [self.outputs[child.name] for child in ete_node.get_children()]
                children_embeddings_mlp = torch.cat([out[0] for out in children_outputs], dim=1)
                children_embeddings_linear = torch.cat([out[1] for out in children_outputs], dim=1)
                embedding_mlp = self.nodes[ete_node.name](children_embeddings_mlp, children_embeddings_linear)
                embedding_linear = self.nodes[ete_node.name].get_node_embedding_linear()
                self.outputs[ete_node.name] = (embedding_mlp, embedding_linear)
                self.total_l0_reg += self.nodes[ete_node.name].get_l0_reg()

        # Pass the tree out through the final out layer
        logits = self.out_layer(self.outputs[self.tree.ete_tree.name][0])

        return logits
    
    def get_total_l0_reg(self):
        return self.total_l0_reg