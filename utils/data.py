import pickle
import sys

import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.common import DotDict
from utils.solver import Solver

SCALING_FACTOR = 1_000

VEHICLE_OPTIONS = {
    "10": (4, 30),
    "20": (5, 40)
}


def store_pickle(obj, file):
    """
    Stores the given object in a pickle file.
    :param obj: object to store
    :param file: location of pickle file
    :return: None
    """
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file):
    """
    Loads the given pickle file.
    :param file: location of pickle file
    :return: Any
    """
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_errors(results):
    return [inst for inst in results if not inst.get('routes')]


def get_vehicle_config(num_nodes):
    """
    Returns the vehicle configuration for the given number of nodes.
    :param num_nodes: number of nodes in the instance
    :return: (num_vehicles, vehicle_capacity)
    """
    return VEHICLE_OPTIONS[str(num_nodes)]


def generate_vrp_instance(num_nodes, random_depot_location=True, demand_low=1, demand_high=10):
    """
    Generates a random instance of the CVRP.
    :param num_nodes: number of nodes to create
    :param bool random_depot_location: depot location is randomly selected if True, otherwise fixed to (0.5, 0.5)
    :param demand_low: demand lower bound
    :param demand_high: demand upper bound
    :return: (num_nodes, 3) locations and demands of the nodes
    """
    locations = np.random.rand(num_nodes, 2)
    demands = np.random.randint(demand_low, demand_high, num_nodes)

    if not random_depot_location:
        locations[0] = [0.5, 0.5]

    demands[0] = 0

    return np.concatenate((locations, demands.reshape(-1, 1)), axis=1, dtype=np.float32)


def create_data_model(locations, demands, vehicle_capacity, num_vehicles, depot_index=0, scaling_factor=SCALING_FACTOR):
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
    data_model['distance_matrix'] = np.round(dist_matrix, 0).astype(int).tolist()
    data_model['demands'] = demands.astype(int).tolist()
    data_model['vehicle_capacities'] = np.full(num_vehicles, vehicle_capacity).astype(int).tolist()
    data_model['num_vehicles'] = int(num_vehicles)
    data_model['depot'] = int(depot_index)

    assert sum(data_model['demands']) <= sum(data_model['vehicle_capacities']), "Total demand exceeds total capacity."

    return data_model


def solve(instance, time_limit=3):
    """
    Solves the given instance of the CVRP.
    :param instance: (n, 3) instance of the CVRP with locations and demands
    :param int time_limit: time limit for the solver
    :return: (dict, Solver) solution and solver object
    """
    num_vehicles, vehicle_capacity = get_vehicle_config(instance.shape[0])
    data_model = create_data_model(locations=instance[:, :2],
                                   demands=instance[:, 2],
                                   vehicle_capacity=vehicle_capacity,
                                   num_vehicles=num_vehicles)
    vrp_solver = Solver(data_model, time_limit=time_limit, first_solution_strategy='SAVINGS')
    solution = vrp_solver.solve()

    return solution, vrp_solver


def generate_and_solve(num_nodes, random_depot_location=True, time_limit=3):
    """
    Generate and solve an instance of the CVRP.
    :param num_nodes: number of nodes to generate
    :param bool random_depot_location: depot location is randomly selected if True, otherwise fixed to (0.5, 0.5)
    :param int time_limit: time limit for the solver
    :return: dict solution
    """
    instance = generate_vrp_instance(num_nodes, random_depot_location=random_depot_location)
    solution, _solver = solve(instance, time_limit)

    if solution:
        solution['instance'] = instance

        return solution

    # try again
    return generate_and_solve(num_nodes, random_depot_location, time_limit)


def distance_matrix(node_coords):
    """
    Computes the distance matrix for a set of node coordinates.
    :param node_coords: (n, 2) array of node coordinates
    :return: (n, n) distance matrix
    """
    return squareform(pdist(node_coords, metric='euclidean'))


def adjacency_matrix(num_nodes, routes):
    """
    Computes the adjacency matrix for a set of routes.
    :param int num_nodes: number of nodes
    :param dict routes: routes
    :return: (n, n) adjacency matrix
    """
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for path in routes.values():
        path = [0] + list(path) + [0]

        for i in range(len(path) - 1):
            xi, xj = path[i], path[i + 1]

            adj_matrix[xi, xj] = 1
            adj_matrix[xj, xi] = 1

    # remove self connections
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix


def edge_feature_matrix(dist_matrix, k=3):
    """
    Computes the edge feature matrix
    :param dist_matrix: (n, n) distance matrix
    :param k: number of nearest neighbors
    :return: (n, n) edge feature matrix
    """
    edge_matrix = np.zeros_like(dist_matrix)

    # find k-nearst neighbors
    for i in range(len(dist_matrix)):
        knns = np.argsort(dist_matrix[i])[1:k + 1]
        edge_matrix[i, knns] = 1  # neat trick

    # self connections
    np.fill_diagonal(edge_matrix, 2)

    # depot connections
    # TODO: test if this is necessary
    # edge_matrix[1:, 0] = 3

    return edge_matrix


class VRPData:
    @classmethod
    def process_dataset(cls, dataset, **kwargs):
        return [cls.process_one(instance, **kwargs) for instance in dataset]

    @classmethod
    def process_one(cls, solution, k=3, normalize_demand=True):
        instance = solution['instance']
        routes = solution['routes']

        node_features = np.array(instance, copy=True)

        if normalize_demand:
            _num_vehicles, vehicle_capacity = get_vehicle_config(instance.shape[0])
            node_features[:, 2] = node_features[:, 2] / vehicle_capacity  # normalise demand

        dist_matrix = distance_matrix(node_features[:, :2])
        target_matrix = adjacency_matrix(node_features.shape[0], routes)
        edge_feat_matrix = edge_feature_matrix(dist_matrix, k=k)

        data = DotDict({
            'node_features': node_features,
            'dist_matrix': dist_matrix,
            'edge_feat_matrix': edge_feat_matrix,
            'target': target_matrix,
            'routes': routes
        })

        return data

    @classmethod
    def load(cls, results, test_size=0.2, random_state=42, **kwargs):
        errors = get_errors(results)

        if len(errors) > 0:
            print("Errors found in some instances, remove them from the dataset.")
            sys.exit(1)

        processed_dataset = cls.process_dataset(results, **kwargs)
        train, test = train_test_split(processed_dataset, test_size=test_size, random_state=random_state, shuffle=True)
        train, test = VRPDataset(train), VRPDataset(test)

        return DotDict({"train": train, "test": test})


class VRPDataset(Dataset):
    def __init__(self, data: list[DotDict]):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        node_features = torch.tensor(data.node_features, dtype=torch.float32)
        dist_matrix = torch.tensor(data.dist_matrix, dtype=torch.float32)
        edge_feat_matrix = torch.tensor(data.edge_feat_matrix, dtype=torch.int64)
        target = torch.tensor(data.target, dtype=torch.int64)

        features = {"node_features": node_features,
                    "dist_matrix": dist_matrix,
                    "edge_feat_matrix": edge_feat_matrix}

        return features, target

    def __repr__(self):
        return f"VRPDataset(len={len(self.data)})"
