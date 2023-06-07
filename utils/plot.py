import networkx as nx
import numpy as np
import seaborn as sns


def route_to_edge_list(routes):
    offset_routes = np.roll(routes, -1)

    return np.stack((routes, offset_routes), -1)


def matrix_to_edge_list(matrix):
    edge_list = []

    for x_i, row in enumerate(matrix):
        x_js = np.argwhere(row > 0).flatten()
        x_is = np.full(x_js.size, x_i)

        edge_list.extend(zip(x_is, x_js))

    return edge_list


def edge_pred_to_edge_list(pred, threshold=0.25):
    num_nodes = len(pred)

    edge_list = []
    edge_values = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            p = pred[i][j]
            if p > threshold:
                edge_list.append((i, j))
                edge_values.append(p)

    return edge_list, edge_values


def plot_graph(nodes, adj_matrix, ax=None):
    """
    Plots the graph.
    :param nodes: locations of the nodes
    :param adj_matrix: adjacency matrix
    :param ax: matplotlib axis
    :return:
    """
    G = nx.from_numpy_array(adj_matrix)
    pos = dict(enumerate(nodes))

    nx.draw_networkx(G, pos, ax=ax, node_color='lightblue')


def plot_heatmap(nodes, adj_matrix, prediction, ax=None):
    """
    Plots the graph with a heatmap of the edge predictions.
    :param nodes: locations of the nodes
    :param adj_matrix: adjacency matrix
    :param prediction: adjacency matrix prediction
    :param ax: matplotlib axis
    :return:
    """
    G = nx.from_numpy_array(adj_matrix)
    pos = dict(enumerate(nodes))

    edge_list, edge_values = edge_pred_to_edge_list(prediction)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_values, edge_cmap=sns.cm.flare, ax=ax)


def plot_beam_search_tour(nodes, adj_matrix, route, ax=None):
    """
    Plots the graph with the beam search tour.
    :param nodes: locations of the nodes
    :param adj_matrix: adjacency matrix
    :param route: route
    :param ax: matplotlib axis
    :return:
    """
    G = nx.from_numpy_array(adj_matrix)
    pos = dict(enumerate(nodes))

    edge_list = route_to_edge_list(route)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_cmap=sns.cm.flare, ax=ax)
