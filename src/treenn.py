import torch
import torch.nn as nn
from captum.module import (BinaryConcreteStochasticGates,
                           GaussianStochasticGates)


class DeterministicGate(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values
        self.zero_tensor = torch.tensor(0.0).to(self.values.device)
    
    def forward(self, x):
        return x * self.values, self.zero_tensor
    
    def get_gate_values(self):
        return self.values

class TreeNode(nn.Module):
    def __init__(self, name, device,
                 input_dim, output_dim, 
                 gate_type, gate_param):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp_embedding_layer = nn.Sequential(nn.Linear(self.input_dim, self.output_dim), nn.ReLU())
        self.linear_embedding_layer = nn.Linear(self.input_dim, self.output_dim)

        if gate_type == 'gaussian':
            self.gate = GaussianStochasticGates(n_gates=1, mask=torch.tensor([0]).to(device), std=gate_param)
        elif gate_type == 'concrete':
            self.gate = BinaryConcreteStochasticGates(n_gates=1, mask=torch.tensor([0]).to(device), temperature=gate_param)
        elif gate_type == 'deterministic':
            self.gate = DeterministicGate(torch.tensor(gate_param).to(device))
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

        
class TreeNN(nn.Module):
    def __init__(self, device, tree, node_min_dim, 
                 node_dim_func, node_dim_func_param,
                 node_gate_type, node_gate_param, 
                 output_dim, output_depth, output_truncation):
        super().__init__()
        self.device = device
        self.tree = tree
        self.node_min_dim = node_min_dim
        self.node_dim_func = node_dim_func
        self.node_dim_func_param = node_dim_func_param
        self.node_gate_type = node_gate_type
        self.node_gate_param = node_gate_param
        self.output_dim = output_dim
        self.output_depth = output_depth
        self.output_truncation = output_truncation
        self._build_network()
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _build_network(self):
        self.nodes = nn.ModuleDict()
        self.output_nodes = []
        self.embedding_nodes = [] 

        # Define the node dimension function
        def dim_func(x, node_dim_func, node_dim_func_param, depth):
            if node_dim_func == "linear":
                max_depth = self.tree.max_depth
                coeff = node_dim_func_param ** (max_depth - depth)
                return int(coeff * x)
            elif node_dim_func == "const":
                return int(node_dim_func_param)
        
        # Iterate over nodes in ETE Tree and build TreeNode
        for ete_node in reversed(list(self.tree.ete_tree.traverse("levelorder"))):
            if self.tree.depths[ete_node.name] >= self.output_depth or not self.output_truncation:
                # Set the input dimensions
                children_out_dims = [self.nodes[child.name].output_dim for child in ete_node.get_children()] if not ete_node.is_leaf else [1]
                node_in_dim = sum(children_out_dims)

                # Set the output dimensions
                node_out_dim = max(self.node_min_dim, dim_func(len(list(ete_node.leaves())), self.node_dim_func, self.node_dim_func_param, self.tree.depths[ete_node.name]))

                # Build the node
                tree_node = TreeNode(ete_node.name, self.device, 
                                     node_in_dim, node_out_dim, 
                                     self.node_gate_type, self.node_gate_param)
                self.nodes[ete_node.name] = tree_node
                
                # Add the node to the output nodes if necessary
                if self.tree.depths[ete_node.name] <= self.output_depth:
                    self.output_nodes.append(ete_node.name)

        # Build output layer
        output_in_dim = sum([self.nodes[output_node].output_dim for output_node in self.output_nodes])
        self.output_layer = nn.Sequential(nn.BatchNorm1d(output_in_dim), nn.Linear(output_in_dim, self.output_dim))

    def forward(self, x):
        # Initialize a dictionary to store the outputs at each node
        outputs = {}
        self.total_l0_reg = torch.tensor(0.0).to(self.device)

        # Perform a forward pass on the leaves
        input_split = torch.split(x, split_size_or_sections=1, dim=1)
        for leaf_node, input in zip(self.tree.ete_tree.leaves(), input_split):
            embedding_mlp = self.nodes[leaf_node.name](input, input)
            embedding_linear = self.nodes[leaf_node.name].get_node_embedding_linear()
            outputs[leaf_node.name] = (embedding_mlp, embedding_linear)
            self.total_l0_reg += self.nodes[leaf_node.name].get_l0_reg()
            
        # Perform a forward pass on the internal nodes in "topological order"
        for ete_node in reversed(list(self.tree.ete_tree.traverse("levelorder"))):
            if not ete_node.is_leaf and (self.tree.depths[ete_node.name] >= self.output_depth or not self.output_truncation):
                children_outputs = [outputs[child.name] for child in ete_node.get_children()]
                children_embeddings_mlp = torch.cat([output[0] for output in children_outputs], dim=1)
                children_embeddings_linear = torch.cat([output[1] for output in children_outputs], dim=1)
                embedding_mlp = self.nodes[ete_node.name](children_embeddings_mlp, children_embeddings_linear)
                embedding_linear = self.nodes[ete_node.name].get_node_embedding_linear()
                outputs[ete_node.name] = (embedding_mlp, embedding_linear)
                self.total_l0_reg += self.nodes[ete_node.name].get_l0_reg()
                # cleanup memory
                for child in ete_node.get_children():
                    del outputs[child.name]
    
        # Concatenate the outputs of the output nodes
        self.embedding = torch.cat([outputs[output_node][0] for output_node in self.output_nodes], dim=1)

        # Pass the tree output through the final output layer
        preds = self.output_layer(self.embedding)

        return preds
    
    def get_total_l0_reg(self):
        return self.total_l0_reg