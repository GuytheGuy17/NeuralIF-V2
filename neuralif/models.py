import copy
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as pyg
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import aggr
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import tril
from apps.data import graph_to_matrix, augment_features
from neuralif.utils import TwoHop, gershgorin_norm
from scipy.sparse.csgraph import reverse_cuthill_mckee


############################
#          Layers          #
############################
class MLP(nn.Module):
    def __init__(self, width, layer_norm=False, activation="relu", activate_final=False):
        super().__init__()
        width = list(filter(lambda x: x > 0, width))
        assert len(width) >= 2, "Need at least one layer in the network!"

        lls = nn.ModuleList()
        for k in range(len(width)-1):
            lls.append(nn.Linear(width[k], width[k+1], bias=True))
            if k != (len(width)-2) or activate_final:
                if activation == "relu":
                    lls.append(nn.ReLU())
                elif activation == "tanh":
                    lls.append(nn.Tanh())
                elif activation == "leakyrelu":
                    lls.append(nn.LeakyReLU())
                elif activation == "sigmoid":
                    lls.append(nn.Sigmoid())
                else:
                    raise NotImplementedError(f"Activation '{activation}' not implemented")

        if layer_norm:
            lls.append(nn.LayerNorm(width[-1]))
        
        self.m = nn.Sequential(*lls)

    def forward(self, x):
        return self.m(x)


class ToLowerTriangular(torch_geometric.transforms.BaseTransform):
    def __init__(self, inplace=False):
        self.inplace = inplace
        
    def __call__(self, data, order=None):
        if not self.inplace:
            data = data.clone()
        
        # TODO: if order is given use that one instead
        if order is not None:
            raise NotImplementedError("Custom ordering not yet implemented...")
        
        # transform the data into lower triag graph
        rows, cols = data.edge_index[0], data.edge_index[1]
        fil = cols <= rows
        l_index = data.edge_index[:, fil]
        edge_embedding = data.edge_attr[fil]
        
        data.edge_index, data.edge_attr = l_index, edge_embedding
        return data

class GraphNet(nn.Module):
    # Follows roughly the outline of torch_geometric.nn.MessagePassing()
    # As shown in https://github.com/deepmind/graph_nets
    # Here is a helpful python implementation:
    # https://github.com/NVIDIA/GraphQSat/blob/main/gqsat/models.py
    # Also allows multirgaph GNN via edge_2_features 
    def __init__(self, node_features, edge_features, global_features=0, hidden_size=0,
                 aggregate="mean", activation="relu", skip_connection=False, edge_features_out=None):
        
        super().__init__()
        
        # different aggregation functions
        if aggregate == "sum":
            self.aggregate = aggr.SumAggregation()
        elif aggregate == "mean":
            self.aggregate = aggr.MeanAggregation()
        elif aggregate == "max":
            self.aggregate = aggr.MaxAggregation()
        elif aggregate == "softmax":
            self.aggregate = aggr.SoftmaxAggregation(learn=True)
        else:
            raise NotImplementedError(f"Aggregation '{aggregate}' not implemented")
        
        self.global_aggregate = aggr.MeanAggregation()
        
        add_edge_fs = 1 if skip_connection else 0
        edge_features_out = edge_features if edge_features_out is None else edge_features_out
        
        # Graph Net Blocks (see https://arxiv.org/pdf/1806.01261.pdf)
        self.edge_block = MLP([global_features + (edge_features + add_edge_fs) + (2 * node_features), 
                               hidden_size,
                               edge_features_out],
                              activation=activation)
        
        self.node_block = MLP([global_features + edge_features_out + node_features,
                               hidden_size,
                               node_features],
                              activation=activation)
        
        # optional set of blocks for global GNN
        self.global_block = None    
        if global_features > 0:
            self.global_block = MLP([edge_features_out + node_features + global_features, 
                                     hidden_size,
                                     global_features],
                                    activation=activation)
    def forward(self, x, edge_index, edge_attr, g=None):
        row, col = edge_index
    
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

    # Prepare inputs for the edge block
        sender_features = x[row]
        receiver_features = x[col]

        edge_inputs = [sender_features, receiver_features, edge_attr]

        if self.global_block is not None:
            assert g is not None, "Need global features for global block"
        
        # Explicitly expand the global features to match the number of edges.
        # This prevents unintended broadcasting that can cause memory to explode.
            num_edges = sender_features.shape[0]
            expanded_g = g.expand(num_edges, -1) 
            edge_inputs.insert(0, expanded_g)

    # Update edge features
        edge_embedding = self.edge_block(torch.cat(edge_inputs, dim=1))
    
    # Aggregate edge features for node updates
        aggregation = self.aggregate(edge_embedding, row, dim_size=x.shape[0])
    
    # Prepare inputs for the node block
        node_inputs = [x, aggregation]
        if self.global_block is not None:
        # Also expand globals for the node update
            num_nodes = x.shape[0]
            expanded_g_nodes = g.expand(num_nodes, -1)
            node_inputs.insert(0, expanded_g_nodes)
        
    # Update node features
        node_embeddings = self.node_block(torch.cat(node_inputs, dim=1))
    
        if self.global_block is not None:
        # Aggregate features for global update
            edge_aggr_global = self.global_aggregate(edge_embedding, dim_size=1)
            node_aggr_global = self.global_aggregate(node_embeddings, dim_size=1)

            global_inputs = [g, node_aggr_global, edge_aggr_global]
            global_embeddings = self.global_block(torch.cat(global_inputs, dim=1))
        
            return edge_embedding, node_embeddings, global_embeddings
    
        else:
            return edge_embedding, node_embeddings, None
        
    

class MP_Block(nn.Module):
    # L@L.T matrix multiplication graph layer
    # Aligns the computation of L@L.T - A with the learned updates
    def __init__(self, last, edge_features_in, edge_features_hidden, node_features, global_features, hidden_size, **kwargs) -> None:
        super().__init__()
        
        # Determine aggregation functions
        if "aggregate" in kwargs and kwargs["aggregate"] is not None:
            aggr = kwargs["aggregate"] if len(kwargs["aggregate"]) == 2 else kwargs["aggregate"] * 2
        else:
            aggr = ["mean", "sum"]
        
        act = kwargs.get("activation", "relu")
        
        # Determine the final output feature size for this block
        edge_features_out_final = 1 if last else edge_features_hidden
        
        # GraphNet for the lower triangular part
        self.l1 = GraphNet(
            node_features=node_features, 
            edge_features=edge_features_in,
            global_features=global_features,
            hidden_size=hidden_size, 
            aggregate=aggr[0], 
            activation=act, 
            edge_features_out=edge_features_hidden
        )

        # GraphNet for the upper triangular part
        self.l2 = GraphNet(
            node_features=node_features, 
            edge_features=edge_features_hidden,
            global_features=global_features,
            hidden_size=hidden_size, 
            aggregate=aggr[1], 
            activation=act, 
            edge_features_out=edge_features_out_final
        )
    
    def forward(self, x, edge_index, edge_attr, global_features):
        # Process the lower triangular part
        edge_embedding, node_embeddings, global_features = self.l1(x, edge_index, edge_attr, g=global_features)
        
        # Flip edge indices to process the upper triangular part
        edge_index_T = torch.stack([edge_index[1], edge_index[0]], dim=0)
        
        # Process the upper triangular part
        edge_embedding, node_embeddings, global_features = self.l2(node_embeddings, edge_index_T, edge_embedding, g=global_features)
        
        return edge_embedding, node_embeddings, global_features

############################
#         Networks         #
############################
class NeuralPCG(nn.Module):
    def __init__(self, **kwargs):
        # NeuralPCG follows the Encoder-Process-Decoder architecture
        super().__init__()
        
        # Network hyper-parameters
        self._latent_size = kwargs["latent_size"]
        self._num_layers = 2
        self._message_passing_steps = kwargs["message_passing_steps"]
        
        # NeuralPCG uses constant number of features for input and output
        self._node_features = 1
        self._edge_features = 1

        # Pre-network transformations
        self.transforms = None
        
        # Encoder - Process - Decoder architecture
        self.encoder_nodes = MLP([self._node_features] + [self._latent_size] * self._num_layers)
        self.encoder_edges = MLP([self._edge_features] + [self._latent_size] * self._num_layers)
        
        # decoder do not have a layer norm
        self.decoder_edges = MLP([self._latent_size] * self._num_layers + [1])
        
        # message passing layers
        self.message_passing = nn.ModuleList([GraphNet(self._latent_size, self._latent_size,
                                                       hidden_size=self._latent_size,
                                                       aggregate="mean")
                                              for _ in range(self._message_passing_steps)])

    def forward(self, data):
        if self.transforms:
            data = self.transforms(data)
        
        x_nodes, x_edges, edge_index = data.x, data.edge_attr, data.edge_index
        
        # save diag elements for later
        diag_idx = edge_index[0] == edge_index[1]
        diag_values = data.edge_attr[diag_idx].clone()

        latent_edges = self.encoder_edges(x_edges)
        latent_nodes = self.encoder_nodes(x_nodes)

        for message_passing_layer in self.message_passing:
            latent_edges, latent_nodes, _ = message_passing_layer(latent_nodes, edge_index, latent_edges)

        # Convert to lower triangular part of a matrix
        decoded_edges = self.decoder_edges(latent_edges)
        
        return self.transform_output_matrix(diag_idx, diag_values, x_nodes, edge_index, decoded_edges)
        
    def transform_output_matrix(self, diag_idx, diag_vals, node_x, edge_index, edge_values):
        # set the diagonal elements
        # the diag element gets duplicated later so we need to divide by 2
        edge_values[diag_idx] = 0.5 * torch.sqrt(diag_vals)
        size = node_x.size()[0]
        # In class NeuralPCG, method transform_output_matrix
        if torch.is_inference_mode_enabled():
        # Symmetrize and get lower triangular part just like in the training 'else' block
            transpose_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            sym_value = torch.cat([edge_values, edge_values]).squeeze()
            sym_index = torch.cat([edge_index, transpose_index], dim=1)
        
            mask = sym_index[0] <= sym_index[1]
            t = torch.sparse_coo_tensor(sym_index[:, mask], sym_value[mask], size=(size, size))
            t = t.coalesce()

        # Return the PyTorch sparse tensor and its transpose
            return t, t.T, None
        else:
            # symmetrize the output by stacking things!
            transpose_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            
            sym_value = torch.cat([edge_values, edge_values])
            sym_index = torch.cat([edge_index, transpose_index], dim=1)
            sym_value = sym_value.squeeze()
            
            # return only lower triangular part
            m = sym_index[0] <= sym_index[1]
            
            t = torch.sparse_coo_tensor(sym_index[:, m], sym_value[m],
                                        size=(size, size))
            t = t.coalesce()
            
            return t, t.T, None


class PreCondNet(nn.Module):
    # BASELINE MODEL
    # No splitting of the matrix into lower and upper part for alignment
    # Used for the ablation study
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self.global_features = kwargs["global_features"]
        self.latent_size = kwargs["latent_size"]
        # node features are augmented with local degree profile
        self.augment_node_features = kwargs["augment_nodes"]
        
        num_node_features = 8 if self.augment_node_features else 1
        message_passing_steps = kwargs["message_passing_steps"]
        
        self.skip_connections = kwargs["skip_connections"]
        
        # create the layers
        self.mps = torch.nn.ModuleList()
        for l in range(message_passing_steps):
            self.mps.append(GraphNet(num_node_features,
                                     edge_features=1,
                                     hidden_size=self.latent_size,
                                     skip_connection=(l > 0 and self.skip_connections)))

    def forward(self, data):
        
        if self.augment_node_features:
            data = augment_features(data)
        
        # get the input data
        edge_embedding = data.edge_attr
        node_embedding = data.x
        edge_index = data.edge_index
        # add remaining self loops
        a_edges = edge_embedding.clone()
        
        # compute the output of the network
        for i, layer in enumerate(self.mps):
            if i != 0 and self.skip_connections:
                edge_embedding = torch.cat([edge_embedding, a_edges], dim=1)
            
            edge_embedding, node_embedding, _ = layer(node_embedding, edge_index, edge_embedding)
        
        # transform the output into a matrix
        return self.transform_output_matrix(node_embedding, edge_index, edge_embedding)
    
    def transform_output_matrix(self, node_x, edge_index, edge_values):
        # force diagonal to be positive (via activation function)
        diag = edge_index[0] == edge_index[1]
        edge_values[diag] = torch.sqrt(torch.exp(edge_values[diag]))
        edge_values = edge_values.squeeze()
        
        size = node_x.size()[0]
        
        if torch.is_inference_mode_enabled():
        # Symmetrize and get lower triangular part just like in the training 'else' block
            transpose_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            sym_value = torch.cat([edge_values, edge_values])
            sym_index = torch.cat([edge_index, transpose_index], dim=1)

            m_mask = sym_index[0] <= sym_index[1]
        
            t = torch.sparse_coo_tensor(sym_index[:, m_mask], sym_value[m_mask], 
                                    size=(size, size))
            t = t.coalesce()
        
        # Return the PyTorch sparse tensor and its transpose
            return t, t.T, None
        
        else:
            # symmetrize the output
            # we basicially just stack the indices of the matrix and it's transpose
            # when coalesce the result, these results get summed up.
            transpose_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            
            sym_value = torch.cat([edge_values, edge_values])
            sym_index = torch.cat([edge_index, transpose_index], dim=1)
            
            # find lower triangular indices
            m = sym_index[0] <= sym_index[1]
            
            # return only lower triangular part of the data
            t = torch.sparse_coo_tensor(sym_index[:, m], sym_value[m], 
                                        size=(size, size))
            
            # take care of duplicate values (to force the output to be symmetric)
            t = t.coalesce()
            
            return t, None, None


class NeuralIF(nn.Module):
    # Neural Incomplete factorization
    def __init__(self, drop_tol=0, **kwargs) -> None:
        super().__init__()
        
        self.global_features = kwargs["global_features"]
        self.latent_size = kwargs["latent_size"]
        self.augment_node_features = kwargs["augment_nodes"]
        
        num_node_features = 8 if self.augment_node_features else 1
        message_passing_steps = kwargs["message_passing_steps"]
        
        edge_features = kwargs.get("edge_features", 1)
        
        self.skip_connections = kwargs["skip_connections"]
        # In NeuralIF.__init__
        self.mps = torch.nn.ModuleList()
        for l in range(message_passing_steps):
    # Determine the input edge dimension for the current block
    # The first block takes 1 feature. 
            input_edge_features = 1 if l == 0 else edge_features + 1

            self.mps.append(MP_Block(
                skip_connections=self.skip_connections, 
                first=l==0,
                last=l==(message_passing_steps-1),
                edge_features_in=input_edge_features, 
                edge_features_hidden=edge_features,
                node_features=num_node_features,
                global_features=self.global_features,
                hidden_size=self.latent_size,
                activation=kwargs["activation"],
                aggregate=kwargs["aggregate"]
            ))
        
        self.node_decoder = MLP([num_node_features, self.latent_size, 1]) if kwargs.get("decode_nodes") else None
        self.normalize_diag = kwargs.get("normalize_diag", False)
        self.diag_aggregate = aggr.SumAggregation()
        self.graph_norm = pyg.norm.GraphNorm(num_node_features) if kwargs.get("graph_norm") else None
        self.tau = drop_tol
        self.two = kwargs.get("two_hop", False)
        
# In neuralif/models.py, inside the NeuralIF class
        # In your neuralif/models.py file, inside the NeuralIF class

    def forward(self, data):
        if self.augment_node_features:
            data = augment_features(data, skip_rhs=True)
    
        if self.two: 
            data = torch_geometric.transforms.TwoHop()(data)
        
        data = ToLowerTriangular()(data)
    
        edge_embedding, l_index, node_embedding = data.edge_attr, data.edge_index, data.x
    
        a_edges = edge_embedding.clone()
        if a_edges.dim() == 1:
            a_edges = a_edges.view(-1, 1)
    
        global_features = None
        if self.global_features > 0:
            global_features = torch.zeros((1, self.global_features), device=data.x.device)

    # --- Corrected Gradient Checkpointing Loop ---
        for i, layer in enumerate(self.mps):
            if i > 0 and self.skip_connections:
            # Directly update the main edge_embedding variable before checkpointing
                edge_embedding = torch.cat([edge_embedding, a_edges], dim=1)
        
        # Pass the main variables directly to the checkpoint function
            edge_embedding, node_embedding, global_features = checkpoint(
                layer, node_embedding, l_index, edge_embedding, global_features, use_reentrant=False
            )
    
        return self.transform_output_matrix(node_embedding, l_index, edge_embedding, a_edges)
    def transform_output_matrix(self, node_x, edge_index, edge_values, a_edges):
    # Clone the tensor to avoid in-place modification and to save the original output
        output_edge_values = edge_values.clone()
    
        diag = edge_index[0] == edge_index[1]
    
        if self.normalize_diag:
            # Ensure the diagonal is positive and normalized
            if a_edges.dim() > 1:
                a_edges = a_edges.squeeze() # Ensure a_edges is 1D for this part
            a_diag = a_edges[diag]
        
            square_values = torch.pow(output_edge_values, 2)
            aggregated = self.diag_aggregate(square_values, edge_index[0])
        
        # This line creates a new tensor, so it is safe.
            output_edge_values = torch.sqrt(a_diag[edge_index[0]]) * output_edge_values / torch.sqrt(aggregated[edge_index[0]])
        else:
            predicted_diag = torch.sqrt(torch.exp(output_edge_values[diag]))
            output_edge_values[diag] = predicted_diag + 1e-4

    
        node_output = self.node_decoder(node_x).squeeze() if self.node_decoder is not None else None
    
        if torch.is_inference_mode_enabled():
            m = torch.sparse_coo_tensor(edge_index, output_edge_values.squeeze(),
                                     size=(node_x.size()[0], node_x.size()[0]))
            return m, m.T, node_output
        else:
            t = torch.sparse_coo_tensor(edge_index, output_edge_values.squeeze(),
                                     size=(node_x.size()[0], node_x.size()[0]))
        
        # Calculate the L1 penalty
            l1_penalty = torch.sum(torch.abs(edge_values)) / len(edge_values)
        
            return t, l1_penalty, node_output

class NeuralIFWithRCM(NeuralIF):
    """
    Extends NeuralIF with a one-shot Reverse Cuthill-McKee permutation
    baked into the forward pass.
    """
    def forward(self, data):
    # Compute RCM permutation
        with torch.no_grad():
            n = data.x.size(0)
            A_csr = to_scipy_sparse_matrix(data.edge_index, data.edge_attr, n).tocsr()
            perm_np = reverse_cuthill_mckee(A_csr) 
            p = torch.as_tensor(perm_np.copy(), dtype=torch.long, device=data.x.device)
            invp = torch.empty_like(p)
            invp[p] = torch.arange(p.numel(), device=p.device)

    # Permute the graph data object
        data.x = data.x[p]
        if hasattr(data, 'batch'):
            data.batch = data.batch[p]
        data.edge_index = invp[data.edge_index]

    #  Call the base NeuralIF.forward on the reordered graph
        out1, out2, node_out = super().forward(data)

    # Invert permutation on outputs to return them to the original ordering
        def invert_tensor(M):
            if isinstance(M, torch.Tensor) and M.is_sparse:
            # For sparse tensors, permute the indices
                idx, vals = M.coalesce().indices(), M.coalesce().values()
                new_idx = p[idx] 
                return torch.sparse_coo_tensor(new_idx, vals, M.shape, device=M.device)
            elif isinstance(M, torch.Tensor) and node_out is not None:
                pass 
            return M

        L = invert_tensor(out1)
        U_or_reg = invert_tensor(out2) if isinstance(out2, torch.Tensor) else out2
    
    # The node_out tensor should be permuted back using the inverse permutation
        if isinstance(node_out, torch.Tensor):
            node_out = node_out[invp]

        return L, U_or_reg, node_out


class LearnedLU(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        
        self.global_features = kwargs["global_features"]
        self.augment_node_features = kwargs["augment_nodes"]
        
        num_node_features = 8 if self.augment_node_features else 1
        
        message_passing_steps = kwargs["message_passing_steps"]
        self.skip_connections = kwargs["skip_connections"]
        self.layers = nn.ModuleList()
        
        # use a smooth activation function for the diagonal during training
        self.smooth_activation = kwargs.get("smooth_activation", True)
        self.epsilon = kwargs.get("epsilon", 0.001)
        
        num_edge_features = 32
        hidden_size = 32
        
        for l in range(message_passing_steps):
            first_layer = l == 0
            last_layer = l == (message_passing_steps - 1)
            
            self.layers.append(
                GraphNet(
                    skip_connection=(l != 0 and self.skip_connections),
                    edge_features=2 if first_layer else num_edge_features,
                    edge_features_out=1 if last_layer else num_edge_features,
                    hidden_size=hidden_size,
                    node_features=num_node_features,
                    global_features=self.global_features
                )
            )
    
    def forward(self, data):
        a_edges = data.edge_attr.clone()

        if self.augment_node_features:
            data = augment_features(data)
        
        # add remaining self loops
        data.edge_index, data.edge_attr = torch_geometric.utils.add_remaining_self_loops(data.edge_index, data.edge_attr)
        
        edge_embedding = data.edge_attr
        node_embedding = data.x
        edge_index = data.edge_index
        
        
        # add positional encoding features
        row, col = data.edge_index
        lower_mask = row > col
        upper_mask = row < col
        additional_edge_feature = torch.zeros_like(a_edges)
        additional_edge_feature[lower_mask] = -1
        additional_edge_feature[upper_mask] = 1
        edge_embedding = torch.cat([edge_embedding, additional_edge_feature], dim=1)
        
        if self.global_features > 0:
            global_features = torch.zeros((1, self.global_features), device=data.x.device, requires_grad=False)
        else:
            global_features = None
        
        for i, layer in enumerate(self.layers):
            if i != 0 and self.skip_connections:
                edge_embedding = torch.cat([edge_embedding, a_edges], dim=1)
                
            edge_embedding, node_embedding, global_features = layer(node_embedding, edge_index, edge_embedding, global_features)
        
        return self.transform_output_matrix(a_edges, node_embedding, edge_index, edge_embedding)
    
    def transform_output_matrix(self, a_edges, node_x, edge_index, edge_values):
        """
        Transform the output into L and U matrices.

        Parameters:
            a_edges (Tensor): Original edge attributes.
            node_x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            edge_values (Tensor): Edge values.
            tolerance (float): Tolerance for small values.

        Returns:
            tuple: Lower and upper matrices, and L1 norm.
        """
        
        @torch.no_grad()
        def step_activation(x, eps=0.05):
            # activation function to enfore the diagonal to be non-zero
            # - replace small values with epsilon
            # - replace zeros with epsilon
            s = torch.where(torch.abs(x) > eps, x, torch.sign(x) * eps)
            return torch.where(s == 0, eps, s)
            
        def smooth_activation(x, eps=0.05):
            return x * (1 + torch.exp(-torch.abs((4 / eps) * x) + 2))
        
        # create masks to split the edge values
        lower_mask = edge_index[0] >= edge_index[1]
        upper_mask = edge_index[0] <= edge_index[1]
        diag_mask = edge_index[0] == edge_index[1]
        
        # create values and indices for lower part
        lower_indices = edge_index[:, lower_mask]
        lower_values = edge_values[lower_mask][:, 0].squeeze()
        
        # create values and indices for upper part
        upper_indices = edge_index[:, upper_mask]
        upper_values = edge_values[upper_mask][:, -1].squeeze()
        
        # enforce diagonal to be unit valued for the upper part
        upper_values[diag_mask[upper_mask]] = 1
        
        # appy activation function to lower part
        if torch.is_inference_mode_enabled():
            lower_values[diag_mask[lower_mask]] = step_activation(lower_values[diag_mask[lower_mask]], eps=self.epsilon)
        elif self.smooth_activation:
            lower_values[diag_mask[lower_mask]] = smooth_activation(lower_values[diag_mask[lower_mask]], eps=self.epsilon)
        
        # construct L and U matrix
        n = node_x.size()[0]
        
        # convert to lower and upper matrices
        lower_matrix = torch.sparse_coo_tensor(lower_indices, lower_values.squeeze(), size=(n, n))
        upper_matrix = torch.sparse_coo_tensor(upper_indices, upper_values.squeeze(), size=(n, n))
        
        if torch.is_inference_mode_enabled():
            # In inference mode, we return the matrices directly
            # and do not compute the L1 norm
            return lower_matrix, upper_matrix, None
        
        else:
            # min diag element as a regularization term
            bound = torch.min(torch.abs(lower_values[diag_mask[lower_mask]]))
            
            return (lower_matrix, upper_matrix), bound, None


############################
#         HELPERS          #
############################
