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
    "10": (4, 25),
    "20": (5, 40)
}


def load_pickle(file):
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


def generate_vrp_instance(num_nodes, depot_location=None, demand_low=1, demand_high=10):
    """
    Generates a random instance of the CVRP.
    :param num_nodes: number of nodes to create
    :param depot_location: location of the depot node
    :param demand_low: demand lower bound
    :param demand_high: demand upper bound
    :return: (num_nodes, 3) locations and demands of the nodes
    """
    if depot_location is None:
        depot_location = [0.5, 0.5]

    locations = np.random.rand(num_nodes, 2)
    demands = np.random.randint(demand_low, demand_high + 1, num_nodes)

    locations[0] = depot_location
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


def solve(instance):
    """
    Solves the given instance of the CVRP.
    :param instance: (n, 3) instance of the CVRP with locations and demands
    :return: (dict, Solver) solution and solver object
    """
    num_vehicles, vehicle_capacity = get_vehicle_config(instance.shape[0])
    data_model = create_data_model(locations=instance[:, :2],
                                   demands=instance[:, 2],
                                   vehicle_capacity=vehicle_capacity,
                                   num_vehicles=num_vehicles)
    vrp_solver = Solver(data_model, time_limit=3, first_solution_strategy='SAVINGS')
    solution = vrp_solver.solve()

    return solution, vrp_solver


def generate_and_solve(num_nodes):
    """
    Generate and solve an instance of the CVRP.
    :param num_nodes: number of nodes to generate
    :return: dict solution
    """
    instance = generate_vrp_instance(num_nodes)
    solution, _solver = solve(instance)

    if solution:
        solution['instance'] = instance

        return solution

    # try again
    return generate_and_solve(num_nodes)


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

    return edge_matrix


class Data:

    @classmethod
    def process_dataset(cls, raw_dataset, k=3):
        return [cls.process_one(instance, k=k) for instance in raw_dataset]

    @classmethod
    def process_one(cls, instance, k=3):
        node_demands, routes = instance

        node_demands = np.array(node_demands)

        coords = np.array(node_demands)[:, :2]
        demand = np.array(node_demands)[:, 2]
        dist_matrix = distance_matrix(coords)
        target_matrix = adjacency_matrix(coords.shape[0], routes)
        edge_matrix = edge_feature_matrix(dist_matrix, k=k)

        data = DotDict({
            'coords': coords,
            'demand': demand,
            'distance_matrix': dist_matrix,
            'edge_feat_matrix': edge_matrix,
            'target': target_matrix,
            'routes': routes
        })

        return data

    @classmethod
    def load(cls, results, test_size=0.2, random_state=42):
        errors = get_errors(results)

        if len(errors) > 0:
            print("Errors found in some instances, remove them from the dataset.")
            sys.exit(1)

        dataset = [(result['instance'], result['routes']) for result in results]
        train, test = train_test_split(dataset, test_size=test_size, random_state=random_state, shuffle=True)
        train, test = cls.process_dataset(train), cls.process_dataset(test)

        return train, test


class VRPDataset(Dataset):
    def __init__(self, raw_dataset: list[DotDict]):
        super().__init__()
        self.data = raw_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_instance = self.data[idx]

        coords = torch.tensor(raw_instance.coords, dtype=torch.float32)
        demand = torch.tensor(raw_instance.demand, dtype=torch.float32)
        dist_matrix = torch.tensor(raw_instance.distance_matrix, dtype=torch.float32)
        edge_feat_matrix = torch.tensor(raw_instance.edge_feat_matrix, dtype=torch.int64)
        target = torch.tensor(raw_instance.target, dtype=torch.int64)

        return (coords, demand, dist_matrix, edge_feat_matrix), target

    def __repr__(self):
        return f"VRPDataset(len={len(self.data)})"
