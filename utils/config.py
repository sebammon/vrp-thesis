from utils.common import DotDict


def load_config(**kwargs):
    """
    Loads the configuration for the model.
    :param int hidden_dim: hidden dimension of the model
    :param int node_features: number of node features
    :param int edge_distance_features: number of edge distance features
    :param int edge_types_features: number of edge type features
    :param int num_gcn_layers: number of GCN layers
    :param int num_mlp_layers: number of MLP layers
    :param int dropout: dropout probability
    :return: DotDict
    """
    config = DotDict()

    config.hidden_dim = kwargs.get('hidden_dim', 16)
    config.node_features = kwargs.get('node_features', 3)
    config.edge_distance_features = kwargs.get('edge_distance_features', 1)
    config.edge_types_features = kwargs.get('edge_types_features', 3)
    config.num_gcn_layers = kwargs.get('num_gcn_layers', 5)
    config.num_mlp_layers = kwargs.get('num_mlp_layers', 3)
    config.dropout = kwargs.get('dropout', None)

    return config
