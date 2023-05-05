import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

Instance = namedtuple("Instance", ["coords", "demand", "distance_matrix", "edge_matrix", "target_matrix", "routes"])


class Data:
    @staticmethod
    def distance_matrix(node_coords):
        return squareform(pdist(node_coords, metric='euclidean'))

    @staticmethod
    def target_matrix(num_nodes, routes):
        adj_matrix = np.zeros((num_nodes, num_nodes))

        for path in routes.values():
            # start and end at depot
            path = [0] + list(path) + [0]

            for i in range(len(path) - 1):
                xi, xj = path[i], path[i + 1]

                adj_matrix[xi, xj] = 1
                adj_matrix[xj, xi] = 1

        # remove self connections
        np.fill_diagonal(adj_matrix, 0)

        return adj_matrix

    @staticmethod
    def edge_matrix(distance_matrix, k=3):
        edge_matrix = np.zeros_like(distance_matrix)

        # find k-nearst neighbors
        for i in range(len(distance_matrix)):
            knns = np.argsort(distance_matrix[i])[1:k + 1]
            edge_matrix[i, knns] = 1  # neat trick

        # self connections
        np.fill_diagonal(edge_matrix, 2)

        return edge_matrix

    @classmethod
    def pre_process(cls, raw_dataset, k=3):
        return [cls.process_instance(instance, k=k) for instance in raw_dataset]

    @classmethod
    def process_instance(cls, instance, k=3):
        node_demands, routes = instance

        node_demands = np.array(node_demands)

        coords = np.array(node_demands)[:, :2]
        demand = np.array(node_demands)[:, 2]
        distance_matrix = cls.distance_matrix(coords)
        target_matrix = cls.target_matrix(len(coords), routes)
        edge_matrix = cls.edge_matrix(distance_matrix, k=k)

        data = Instance(coords, demand, distance_matrix, edge_matrix, target_matrix, routes)

        return data

    @classmethod
    def load(cls, filename="vrp_11", test_size=0.2, random_state=42):
        if filename.endswith(".pkl"):
            raise ValueError("Filename must not end with .pkl")

        data_dir = Path(__file__).parent.parent / "data"
        with open(data_dir / f"{filename}.pkl", 'rb') as f:
            instances = pickle.load(f)

        with open(data_dir / f"{filename}_solved.pkl", 'rb') as f:
            solutions = pickle.load(f)

        assert len(instances) == len(solutions), "Number of instances and solutions must be equal"
        assert np.all(
            [solution['instance'] == i for i, solution in enumerate(solutions)]), "Instance numbers must be equal"

        dataset = [(instance, solution['routes']) for instance, solution in zip(instances, solutions)]
        train, test = train_test_split(dataset, test_size=test_size, random_state=random_state, shuffle=True)
        train, test = cls.pre_process(train), cls.pre_process(test)

        return train, test


class VRPDataset(Dataset):
    def __init__(self, raw_dataset: list[Instance]):
        super().__init__()
        self.data = raw_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_instance = self.data[idx]

        coords = torch.tensor(raw_instance.coords, dtype=torch.float32)
        demand = torch.tensor(raw_instance.demand, dtype=torch.float32)
        distance_matrix = torch.tensor(raw_instance.distance_matrix, dtype=torch.float32)
        edge_matrix = torch.tensor(raw_instance.edge_matrix, dtype=torch.float32)
        target_matrix = torch.tensor(raw_instance.target_matrix, dtype=torch.int64)

        instance = Instance(coords, demand, distance_matrix, edge_matrix, target_matrix, None)

        return instance

    def __repr__(self):
        return f"GraphDataset(len={len(self.data)})"
