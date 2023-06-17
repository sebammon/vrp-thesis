import torch
import torch.nn as nn
import torch.nn.functional as F


# == NORMALISATION LAYERS ==
# The normalisation layers are required because the tensors need to
# be transposed for batch normalisation


class EdgeNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        """
        Args:
            e: Edge features (batch x num_nodes x num_nodes x hidden_dim)
        """
        # transpose because batch norm works on the third dim
        e_trans = e.transpose(
            1, 3
        ).contiguous()  # B x hidden_dim x num_nodes x num_nodes
        e_trans_batch_norm = self.batch_norm(e_trans)
        e_batch_norm = e_trans_batch_norm.transpose(
            1, 3
        ).contiguous()  # B x num_nodes x num_nodes x hidden_dim

        return e_batch_norm


class NodeNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        """
        Args:
            x: Node features (batch x num_nodes x hidden_dim)
        """
        # transpose because batch norm works on the second dim
        x_trans = x.transpose(1, 2).contiguous()  # B x hidden_dim x num_nodes
        x_trans_batch_norm = self.batch_norm(x_trans)
        x_batch_norm = x_trans_batch_norm.transpose(
            1, 2
        ).contiguous()  # B x num_nodes x hidden_dim

        return x_batch_norm


class EdgeFeatureLayer(nn.Module):
    """
    W_3 e_ij + W_4 (x_i + x_j) <-- currently the case, but should be: W_3 e_ij + W_4 x_i + W_5 x_j
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim)
        # TODO: Why are not two Linear layers used W_4 and W_5 - does it make a difference?
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch x num_nodes x hidden_dim)
            e: Edge features (batch x num_nodes x num_nodes x hidden_dim)
        """
        Ue = self.U(e)
        Vx = self.V(x)

        # this enables us to make use of broadcasting to get a B x num_nodes x num_nodes x hidden_dim tensor
        Vx_cols = Vx.unsqueeze(
            1
        )  # B x num_nodes x hidden_dim => B x 1 x num_nodes x hidden_dim
        Vx_rows = Vx.unsqueeze(
            2
        )  # B x num_nodes x hidden_dim => B x num_nodes x 1 x hidden_dim

        e_new = Ue + Vx_rows + Vx_cols

        return e_new


class NodeFeatureLayer(nn.Module):
    """
    W_1 x_i + ( sum_j( n_ij * W_2 x_j ) )

    where: n_ij = gate_ij / sum_j ( gate_ij + e )
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.epsilon = 1e-20

    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch x num_nodes x hidden_dim)
            edge_gate: Edge gate run through a sigmoid (batch x num_nodes x num_nodes x hidden_dim)
        """
        Ux = self.U(x)
        Vx = self.V(x)  # B x num_nodes x hidden_dim

        Vx = Vx.unsqueeze(
            1
        )  # B x num_nodes x hidden_dim ==> B x 1 x num_nodes x hidden_dim
        gateVx = edge_gate * Vx

        x_new = Ux + torch.sum(gateVx, dim=2) / (
            self.epsilon + torch.sum(edge_gate, dim=2)
        )  # B x num_nodes x hidden_dim

        return x_new


# == GRAPH LAYER ==


class GraphLayer(nn.Module):
    """
    Graph layer for x_i and e_ij
    """

    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.node_feat = NodeFeatureLayer(hidden_dim)
        self.node_norm = NodeNorm(hidden_dim)
        self.edge_feat = EdgeFeatureLayer(hidden_dim)
        self.edge_norm = EdgeNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch x node_num x hidden_dim)
            e: Edge features (batch x num_nodes x num_nodes x hidden_dim)
        Return:
            x: Aggregated node features (batch x node_num x hidden_dim)
            e: Aggragated edge features (batch x num_nodes x num_nodes x hidden_dim)
        """
        # edges
        e_feat = self.edge_feat(x, e)

        # edge gates
        e_gates = F.sigmoid(e_feat)

        # nodes
        x_feat = self.node_feat(x, e_gates)

        # normalisation
        x_norm = self.node_norm(x_feat)
        e_norm = self.edge_norm(e_feat)

        # dropout
        if self.dropout:
            x_drop = F.dropout(x_norm, p=self.dropout, training=self.training)
            e_drop = F.dropout(e_norm, p=self.dropout, training=self.training)
        else:
            x_drop = x_norm
            e_drop = e_norm

        # activation
        x_act = F.relu(x_drop)
        e_act = F.relu(e_drop)

        # combine
        x_new = x + x_act
        e_new = e + e_act

        return x_new, e_new


# == MLP (Edge predictions) ==


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers):
        super().__init__()

        self.layers = nn.Sequential()

        for i in range(hidden_layers - 1):
            self.layers.add_module(f"lin{i + 1}", nn.Linear(in_dim, in_dim))
            self.layers.add_module(f"relu{i + 1}", nn.ReLU())

        self.layers.add_module("final", nn.Linear(in_dim, out_dim))

    def forward(self, e):
        """
        Args:
            e: Edge features (batch x num_nodes x num_nodes x hidden_dim)
        Returns:
            y: Edge predictions (batch x num_nodes x num_nodes x out_dim)
        """

        return self.layers(e)


# == MAIN NETWORK ==


class GraphNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # configs
        self.hidden_dim = config.hidden_dim
        self.node_features = config.node_features
        self.edge_distance_features = config.edge_distance_features
        self.edge_types_features = config.edge_types_features
        self.num_gcn_layers = config.num_gcn_layers
        self.num_mlp_layers = config.num_mlp_layers
        self.dropout = config.dropout

        # embeddings
        # TODO: Why is bias turned off when in the paper they don't mention anything?
        self.node_feature_embedding = nn.Linear(
            self.node_features, self.hidden_dim, bias=False
        )
        self.distance_embedding = nn.Linear(
            self.edge_distance_features, self.hidden_dim // 2, bias=False
        )
        # TODO: Don't understand the use of the Embedding layer
        # 3 for the special cases 0, 1, 2 (more memory efficient)
        self.edge_feature_embedding = nn.Embedding(
            self.edge_types_features, self.hidden_dim // 2
        )

        # GCN layers
        self.gcn_layers = nn.ModuleList(
            [
                GraphLayer(hidden_dim=self.hidden_dim, dropout=self.dropout)
                for _ in range(self.num_gcn_layers)
            ]
        )

        # edge prediction MLP
        self.mlp_edges = MLP(
            in_dim=self.hidden_dim, out_dim=2, hidden_layers=self.num_mlp_layers
        )

    def forward(self, node_features, distance_matrix, edge_features):
        """
        Args:
            node_features: Node features for each node (batch x num_nodes x 3)
            distance_matrix: Distance matrix between nodes (batch x num_nodes x num_nodes)
            edge_features: Edge connection types (batch x num_nodes x num_nodes)
        """
        # eq 2
        x = self.node_feature_embedding(node_features)  # B x num_nodes x hidden_dim

        # eq 3
        dist_unsqueezed = distance_matrix.unsqueeze(3)  # B x num_nodes x num_nodes x 1
        e_dist = self.distance_embedding(
            dist_unsqueezed
        )  # B x num_nodes x num_nodes x hidden_dim // 2
        e_values = self.edge_feature_embedding(
            edge_features
        )  # B x num_nodes x num_nodes x hidden_dim // 2
        e = torch.cat(
            (e_dist, e_values), dim=3
        )  # B x num_nodes x num_nodes x hidden_dim

        # eq 4 and 5
        for gcn_layer in self.gcn_layers:
            x, e = gcn_layer(
                x, e
            )  # B x num_nodes x hidden_dim, B x num_nodes x num_nodes x hidden_dim

        # eq 6
        y_edge_pred = self.mlp_edges(e)  # B x num_nodes x num_nodes x 2

        return y_edge_pred
