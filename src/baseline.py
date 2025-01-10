import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.output_layer = nn.Linear(in_features // 2, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.output_layer(x)
        return x
    
class MLPWithTree(nn.Module):
    def __init__(self, tree, out_features,):
        super(MLPWithTree, self).__init__()
        
        self.tree = tree
        self.out_features = out_features

        # Build model layers based on tree structure
        self.layers = nn.ModuleList()

        for depth_level in range(tree.max_depth, 1, -1):
            nodes_at_level = [node for node, depth in tree.depths.items() if depth == depth_level]
            next_level_nodes = [node for node, depth in tree.depths.items() if depth == depth_level - 1]

            input_dim = len(nodes_at_level)
            output_dim = len(next_level_nodes)
            
            layer = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU()
            )
            self.layers.append(layer)
        
        
        # Final layer to map to the desired output features
        self.final_layer = nn.Linear(output_dim, self.out_features)
        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


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
    

class MDeep(nn.Module):
    def __init__(self, 
                 tree, 
                 out_features, 
                 num_filter, 
                 window_size, 
                 stride_size, 
                 keep_prob):
        super(MDeep, self).__init__()
        self.tree = tree
        self.keep_prob = keep_prob
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size

        # Create convolutional layers
        self.conv_layers = self._create_conv_layers(num_filter, window_size, stride_size)

        # Compute the input features for the fully connected layers
        fc1_in_features = self._compute_fc_layer_in_features()

        # Create fully connected layers
        self.fc_layers = self._create_fc_layers(fc1_in_features, 64, out_features)

        # Initialize the feature reordering
        self._init_reordering()

    def _create_conv_layers(self, num_filter, window_size, stride_size):
        layers = []
        for i in range(len(num_filter)):
            in_channels = 1 if i == 0 else num_filter[i - 1]
            out_channels = num_filter[i]
            kernel_size = window_size[i]
            stride = stride_size[i]
            padding = (kernel_size - 1) // 2  # Padding to maintain 'same' output size as input

            # Define a sequential block for each convolutional layer
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(p=self.keep_prob)
            )
            layers.append(conv_block)
        return nn.ModuleList(layers)

    def _create_fc_layers(self, in_features, hidden_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_features),
            nn.Linear(hidden_features, out_features)
        )

    def _compute_fc_layer_in_features(self):
        self.eval()
        num_features = len(list(self.tree.ete_tree.leaves()))
        dummy_input = torch.zeros((1, 1, num_features))
        dummy_output = dummy_input
        for conv_layer in self.conv_layers:
            dummy_output = conv_layer(dummy_output)
        return dummy_output.view(dummy_output.size(0), -1).shape[1]
    
    def _init_reordering(self):
        D, _ = self.tree.ete_tree.cophenetic_matrix()
        D = np.array(D)
        rho = 2
        cor_matrix = np.exp(-2 * rho * D)

        def hac(cor):
            def mydist(p1, p2):
                x = int(p1)
                y = int(p2)
                return 1.0 - cor[x, y]

            x = list(range(cor.shape[0]))
            X = np.array(x)
            linked = linkage(np.reshape(X, (len(X), 1)), metric=mydist, method='single')
            result = dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
            indexes = result.get('ivl')
            return [int(i) for i in indexes]

        self.reordered_indices = hac(cor_matrix)

    def forward(self, x):
        # Reorder the features based on the correlation matrix
        x = x[:, self.reordered_indices]
        x = x.unsqueeze(1)

        # Apply convolutional layers sequentially
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc_layers(x)

        return x
    

class DenseWithTree(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 tree_weight, 
                 kernel_initializer, 
                 bias_initializer='zeros', 
                 use_bias=True):
        super(DenseWithTree, self).__init__()
        self.tree_weight = torch.tensor(tree_weight, dtype=torch.float32)
        self.use_bias = use_bias
        
        # Define a single linear layer
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        
        # Initialize weights
        self._initialize_weights(self.linear.weight, kernel_initializer)
        if use_bias:
            self._initialize_weights(self.linear.bias, bias_initializer)


    def forward(self, x):
        # Apply the tree_weight mask to the kernel weight matrix
        weight = self.linear.weight * self.tree_weight.to(self.linear.weight.device)
        
        # Perform matrix multiplication with the modified weight
        output = torch.matmul(x, weight.t())
        
        # Add bias if enabled
        if self.linear.bias is not None:
            output += self.linear.bias

        return output

    def _initialize_weights(self, tensor, initializer_type):
        # Initialize weights to match TensorFlow initializers
        if initializer_type == 'kaiming_normal':
            nn.init.kaiming_normal_(tensor)
        elif initializer_type == 'xavier_uniform':
            nn.init.xavier_uniform_(tensor)
        elif initializer_type == 'zeros':
            nn.init.zeros_(tensor)
        else:
            raise ValueError(f"Unsupported initializer: {initializer_type}")

class DeepBiome(nn.Module):
    def __init__(self, 
                 tree, 
                 out_features, 
                 batch_norm, 
                 dropout, 
                 weight_decay_type, 
                 weight_initial):
        super(DeepBiome, self).__init__()
        
        self.tree = tree
        self.out_features = out_features
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.weight_decay_type = weight_decay_type

        # Initialize tree weights
        self._set_phylogenetic_tree_info()

        # Build model layers based on tree structure
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        input_dim = self.tree_weight_list[0].shape[1]
        
        for i, (tree_w, tree_wn) in enumerate(zip(self.tree_weight_list, self.tree_weight_noise_list)):
            output_dim = tree_w.shape[0]
            
            # Select layer type based on weight_decay_type
            tree_weight = tree_wn if weight_decay_type == 'phylogenetic_tree' else tree_w
            layer = DenseWithTree(input_dim, output_dim, tree_weight, weight_initial)
            
            self.layers.append(layer)
            
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(output_dim))
                
            self.layers.append(nn.ReLU())
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            
            # Update input dimension for next layer
            input_dim = output_dim
        
        # Final layer to map to the desired output features
        self.final_layer = nn.Linear(input_dim, self.out_features)
        
    def _set_phylogenetic_tree_info(self):
        # Get tree layer weights based on `ete4` tree structure
        self.tree_weight_list = []
        self.tree_weight_noise_list = []

        # Create a dictionary to store the parent of each child node
        child_parent = {}
        for node in self.tree.ete_tree.traverse():
            for child in node.children:
                child_parent[child.name] = node.name
        
        # Iterate over the tree depth levels to create weight matrices
        for depth_level in range(1, self.tree.max_depth):
            nodes_at_level = [node for node, depth in self.tree.depths.items() if depth == depth_level]
            next_level_nodes = [node for node, depth in self.tree.depths.items() if depth == depth_level + 1]
            
            weight_matrix = np.zeros((len(nodes_at_level), len(next_level_nodes)))
            noise_matrix = np.full(weight_matrix.shape, 0.01)
            
            for i, node in enumerate(nodes_at_level):
                for j, next_node in enumerate(next_level_nodes):
                    if child_parent[next_node] == node:
                        weight_matrix[i, j] = 1.0
                        noise_matrix[i, j] = 1.0

            self.tree_weight_list.append(weight_matrix)
            self.tree_weight_noise_list.append(noise_matrix)

        # Reverse the order of the weight matrices
        self.tree_weight_list = self.tree_weight_list[::-1]
        self.tree_weight_noise_list = self.tree_weight_noise_list[::-1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        x = self.final_layer(x)
        return x
    

class PhyloConv1D(nn.Module):
    def __init__(self, distances, nb_neighbors, in_channels, out_channels):
        super(PhyloConv1D, self).__init__()
        self.nb_neighbors = nb_neighbors
        self.distances = distances
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=nb_neighbors, stride=nb_neighbors)
        self.relu = nn.ReLU()


    def _gather_target_neighbors(self, data, indices) :
        batch_size, nb_filters, nb_features = data.shape
        _, _, K = indices.shape

        # Expand indices to include the batch and filter dimensions for gathering
        expanded_indices = indices.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Reshape data to (batch_size, nb_filters, nb_features, 1) for correct gathering
        data_expanded = data.unsqueeze(-1).expand(-1, -1, -1, K)

        # Use gather to extract the neighbors
        gathered_data = torch.gather(data_expanded, 2, expanded_indices)

        # Reshape the gathered data to (batch_size, nb_filters, nb_features * K)
        gathered_data = gathered_data.view(batch_size, nb_filters, nb_features * K)

        return gathered_data

    def _top_k(self):
        _, indices = torch.topk(-self.distances, k=self.nb_neighbors, dim=-1)
        return indices


    def forward(self, X, Coord):
        neighbor_indexes = self._top_k().to(X.device)
        Coord = Coord.to(X.device)
        X_phylongb = self._gather_target_neighbors(X, neighbor_indexes)
        Coord_phylongb = self._gather_target_neighbors(Coord, neighbor_indexes).to(X.device)

        X_conv = self.relu(self.conv(X_phylongb))
        C_conv = self.relu(self.conv(Coord_phylongb))

        return X_conv, C_conv

class PhCNN(nn.Module):
    def __init__(self, tree, out_features, nb_filters, nb_neighbors):
        super(PhCNN, self).__init__()
        self.tree = tree
        self.nb_classes = out_features
        self.nb_filters = nb_filters
        self.nb_neighbors = nb_neighbors
        self.coordinates = self._compute_coordinates()

        self.conv_layers = []
        in_channels = 1
        for i in range(len(nb_filters)):
            out_channels = nb_filters[i]
            self.conv_layers.append(PhyloConv1D(distances=None,
                                                nb_neighbors=nb_neighbors[i],
                                                in_channels=in_channels,
                                                out_channels=out_channels))
            in_channels = out_channels
        
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        dense1_in_features = self._compute_fc_layer_in_features()   
        self.dense1 = nn.Linear(in_features=dense1_in_features, out_features=64)
        self.selu = nn.SELU()
        self.dropout = nn.Dropout(p=0.25)
        self.output = nn.Linear(in_features=64, out_features=out_features) 

    def _compute_coordinates(self):
        distances, _ = self.tree.ete_tree.cophenetic_matrix()
        distances = np.sqrt(distances)
        pca = PCA(n_components=min(306, distances.shape[0]))
        coordinates = pca.fit_transform(distances)
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        coordinates = coordinates.unsqueeze(1).transpose(0, 2)
        return coordinates

    def _euclidean_distances(self, X):
        X = X.permute(1, 2, 0)
        distances = torch.cdist(X, X, p=2)
        return distances
    
    def _compute_fc_layer_in_features(self):
        num_features= len(list(self.tree.ete_tree.leaves()))
        dummy_input = torch.randn((1, 1, num_features))
        crd = self.coordinates
        for conv_layer in self.conv_layers:
            distances = self._euclidean_distances(crd)
            conv_layer.distances = distances
            dummy_input, crd = conv_layer(dummy_input, crd)
        dummy_output = self.pool(dummy_input)
        dummy_output = self.flatten(dummy_output)
        return dummy_output.view(dummy_output.size(0), -1).shape[1]

    def forward(self, X):
        X = X.unsqueeze(1)
        crd = self.coordinates

        for conv_layer_module in self.conv_layers:
            distances = self._euclidean_distances(crd)
            conv_layer_module.distances = distances  # Set distances dynamically
            X, crd = conv_layer_module(X, crd)

        X = self.pool(X)
        X = self.flatten(X)
        X = self.dense1(X)
        X = self.selu(X)
        X = self.dropout(X)
        output = self.output(X)

        return output
