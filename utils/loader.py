class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def load_config(hidden_dim=30, node_features=2, edge_weight_features=1, edge_values_features=3, num_gcn_layers=10,
                num_mlp_layers=3):
    config = DotDict()

    config.hidden_dim = hidden_dim
    config.node_features = node_features
    config.edge_weight_features = edge_weight_features
    config.edge_values_features = edge_values_features
    config.num_gcn_layers = num_gcn_layers
    config.num_mlp_layers = num_mlp_layers

    return config
