import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset

from utils.common import DotDict, load_pickle
from utils.solver import Solver

SCALING_FACTOR = 1000

VEHICLE_OPTIONS = {
    "10": (4, 20),
    "20": (5, 30),
    "50": (10, 40),
}


def get_vehicle_config(num_nodes):
    """
    Returns the vehicle configuration for the given number of nodes.
    :param num_nodes: number of nodes in the instance
    :return: (num_vehicles, vehicle_capacity)
    """
    return VEHICLE_OPTIONS[str(num_nodes)]


def generate_instance(num_nodes, demand_low=1, demand_high=10):
    """
    Generates a random instance of the CVRP.
    :param int num_nodes: number of customer nodes to create
    :param int demand_low: demand lower bound
    :param int demand_high: demand upper bound
    :return: (num_nodes, 3) locations and demands of the nodes
    """
    locations = np.random.rand(num_nodes + 1, 2)
    demands = np.random.randint(demand_low, demand_high, num_nodes + 1)

    demands[0] = 0

    return np.concatenate((locations, demands.reshape(-1, 1)), axis=1, dtype=np.float32)


def create_data_model(
    locations,
    demands,
    vehicle_capacity,
    num_vehicles,
    depot_index=0,
    scaling_factor=SCALING_FACTOR,
):
    """
    Creates the data model for the solver.
    :param locations: (n, 2) array of node locations
    :param demands: (n,) array of node demands
    :param vehicle_capacity: Capacity of the vehicles
    :param num_vehicles: number of vehicles to use
    :param depot_index: index of the depot node
    :param scaling_factor: scaling factor for the distance matrix
    :return: data model
    """
    data_model = {}

    # Google OR Tools requires integer values, so we scale the locations
    dist_matrix = distance_matrix(locations) * scaling_factor

    # Note the conversions to int and list
    data_model["distance_matrix"] = np.round(dist_matrix, 0).astype(int).tolist()
    data_model["demands"] = demands.astype(int).tolist()
    data_model["vehicle_capacities"] = (
        np.full(num_vehicles, vehicle_capacity).astype(int).tolist()
    )
    data_model["num_vehicles"] = int(num_vehicles)
    data_model["depot"] = int(depot_index)

    assert sum(data_model["demands"]) <= sum(
        data_model["vehicle_capacities"]
    ), "Total demand exceeds total capacity."

    return data_model


def solve(instance, **kwargs):
    """
    Solves the given instance of the CVRP.
    :param instance: (n, 3) instance of the CVRP with locations and demands
    :keyword int time_limit: time limit for the solver
    :keyword str first_solution_strategy: first solution strategy for the solver
    :return: (dict, Solver) solution and solver object
    """
    num_vehicles, vehicle_capacity = get_vehicle_config(instance.shape[0] - 1)
    data_model = create_data_model(
        locations=instance[:, :2],
        demands=instance[:, 2],
        vehicle_capacity=vehicle_capacity,
        num_vehicles=num_vehicles,
    )
    solver = Solver(data_model, **kwargs)
    solver.solve()

    routes = solver.get_routes()

    return routes, solver


def generate_and_solve(num_nodes, time_limit=3):
    """
    Generate and solve an instance of the CVRP.
    :param num_nodes: number of nodes to generate
    :param int time_limit: time limit for the solver
    :return: dict solution
    """
    instance = generate_instance(num_nodes)
    routes, solver = solve(instance, time_limit=time_limit)

    if len(routes) > 0:
        solution = {
            "instance": instance,
            "routes": routes,
            "vehicle_capacity": solver.data["vehicle_capacities"][0],
        }

        return solution

    # try again
    return generate_and_solve(num_nodes, time_limit)


def distance_from_adj_matrix(batch_targets, batch_dist_matrix):
    """
    Computes the total distance for a batch of adjacency matrices.
    :param batch_targets: Adjacency matrices
    :param batch_dist_matrix: Distance matrices
    :return: (batch_size,) total distance for each adjacency matrix
    """
    return (batch_targets * batch_dist_matrix).sum(-1).sum(-1) / 2


def adj_matrix_from_routes(routes, num_nodes):
    """
    Converts a batch of routes to a batch of adjacency matrices.
    :param routes: Batch of route
    :param num_nodes: Number of nodes
    :return: Batch of adjacency matrices
    """
    routes_rolled = np.roll(routes, -1)
    non_zero_entries = np.stack((routes, routes_rolled), 2)

    matrix = np.zeros((routes.shape[0], num_nodes, num_nodes))

    for i, indices in enumerate(non_zero_entries):
        matrix[i, indices[:, 0], indices[:, 1]] = 1
        matrix[i, indices[:, 1], indices[:, 0]] = 1

    return matrix


def distance_matrix(node_coords):
    """
    Computes the distance matrix for a set of node coordinates.
    :param node_coords: (n, 2) array of node coordinates
    :return: (n, n) distance matrix
    """
    return squareform(pdist(node_coords, metric="euclidean"))


def adjacency_matrix(num_nodes, routes):
    """
    Computes the adjacency matrix for a set of routes.
    :param int num_nodes: number of nodes
    :param routes: (m, n) routes
    :return: (n, n) adjacency matrix
    """
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for path in routes:
        for i in range(len(path) - 1):
            xi, xj = path[i], path[i + 1]

            adj_matrix[xi, xj] = 1
            adj_matrix[xj, xi] = 1

    # remove self connections
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix


def edge_feature_matrix(dist_matrix, k, special_connections=False):
    """
    Computes the edge feature matrix
    :param dist_matrix: (n, n) distance matrix
    :param k: number of nearest neighbors
    :param special_connections: whether to add special connections
    :return: (n, n) edge feature matrix
    """
    edge_matrix = np.zeros_like(dist_matrix)

    if k is None:
        k = len(dist_matrix)

    # find k-nearst neighbors
    for i in range(len(dist_matrix)):
        knns = np.argsort(dist_matrix[i])[1 : k + 1]
        edge_matrix[i, knns] = 1  # neat trick

    if special_connections:
        edge_matrix[:, 0] += 3
        edge_matrix[0, :] += 3

    # self connections
    np.fill_diagonal(edge_matrix, 2)

    return edge_matrix


def load_and_split_dataset(file, test_size=0.2, shuffle=True, random_state=42):
    """
    Loads and splits the dataset.
    :param file: location of the dataset
    :param test_size: train split
    :param shuffle: shuffle the dataset
    :param random_state: random state
    :return: (list, list) train and test split
    """
    dataset = load_pickle(file)
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=test_size, random_state=random_state, shuffle=shuffle
    )

    return train_dataset, test_dataset


def process_datasets(*raw_datasets, **kwargs):
    """
    Loads and processes the datasets.
    :argument list raw_datasets: list of raw datasets
    :keyword int k: number of nearest neighbors
    :return: tuple of datasets
    """

    return tuple(VRPDataset(dataset, **kwargs) for dataset in list(*raw_datasets))


def dataset_class_weight(dataset):
    """
    Computes the class weights for the dataset.
    :param dataset: dataset
    :return: class weights
    """
    targets = np.array([target_edges.detach().numpy() for _, target_edges in dataset])

    class_labels = targets.flatten()

    edge_class_weights = compute_class_weight(
        "balanced", classes=np.unique(class_labels), y=class_labels
    )
    edge_class_weights = torch.tensor(edge_class_weights, dtype=torch.float)

    return edge_class_weights


class VRPDataset(Dataset):
    def __init__(
        self, raw_dataset, k, special_connections=False, normalise_demand=True
    ):
        super().__init__()
        self.data = []
        self.k = k
        self.special_connections = special_connections
        self.normalise_demand = normalise_demand

        self.process_dataset(raw_dataset)

    def __process_one(self, instance):
        routes = instance["routes"]
        node_features = instance["instance"]
        vehicle_capacity = instance["vehicle_capacity"]

        if self.normalise_demand:
            node_features[:, 2] = node_features[:, 2] / vehicle_capacity

        dist_matrix = distance_matrix(node_features[:, :2])
        edge_feat_matrix = edge_feature_matrix(
            dist_matrix, k=self.k, special_connections=self.special_connections
        )
        target_matrix = adjacency_matrix(node_features.shape[0], routes)

        return DotDict(
            {
                "node_features": node_features,
                "dist_matrix": dist_matrix,
                "edge_feat_matrix": edge_feat_matrix,
                "target": target_matrix,
                "routes": routes,
            }
        )

    def process_dataset(self, results):
        for instance in results:
            data = self.__process_one(instance)
            self.data.append(data)

    def class_weights(self):
        return dataset_class_weight(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        num_vehicles = torch.tensor(len(data.routes), dtype=torch.int64)
        node_features = torch.tensor(data.node_features, dtype=torch.float32)
        dist_matrix = torch.tensor(data.dist_matrix, dtype=torch.float32)
        edge_feat_matrix = torch.tensor(data.edge_feat_matrix, dtype=torch.int64)
        target = torch.tensor(data.target, dtype=torch.int64)

        features = {
            "node_features": node_features,
            "dist_matrix": dist_matrix,
            "edge_feat_matrix": edge_feat_matrix,
            "num_vehicles": num_vehicles,
        }

        return features, target
