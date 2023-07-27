import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.beam_search import BeamSearch
from utils.common import DotDict
from utils.data import adj_matrix_from_routes, distance_from_adj_matrix


def eval_model(batch_node_features, batch_dist_matrix, batch_edge_features, model):
    """
    Evaluates the model on the given batch.
    :param batch_node_features: node features of the batch
    :param batch_dist_matrix: distance matrix of the batch
    :param batch_edge_features: edge features of the batch
    :param model: model to evaluate
    :return: predictions of the model (after applying softmax)
    """
    model.eval()

    with torch.no_grad():
        preds = model(batch_node_features, batch_dist_matrix, batch_edge_features)
        preds = F.softmax(preds, dim=3)

        return preds


def get_metrics(targets, predictions):
    """
    Computes the metrics for the given targets and predictions.
    :param targets: (batch_size, num_nodes, num_nodes)
    :param predictions: (batch_size, num_nodes, num_nodes)
    :return: DotDict with metrics
    """
    acc = accuracy_score(targets.flatten(), predictions.flatten())
    bal_acc = balanced_accuracy_score(
        targets.flatten(), predictions.flatten(), adjusted=True
    )
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        targets.flatten(), predictions.flatten(), average="binary"
    )
    return DotDict(
        {
            "acc": acc,
            "bal_acc": bal_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
    )


def is_tour_over_capacity(tour, demand):
    loads = np.take(demand, tour)
    running_load = 0

    for j in range(len(loads)):
        running_load += loads[j]

        if tour[j] == 0 or j == len(loads) - 1:
            if np.round(running_load, 2) > 1.0:
                return True
            running_load = 0

    return False


def _check_valid(tour, demand, num_nodes):
    count = np.bincount(tour, minlength=num_nodes)

    over_capacity = is_tour_over_capacity(tour, demand)
    all_nodes_visited = np.all(count[1:] == 1)

    return all_nodes_visited and not over_capacity


def check_valid_tours(tours, demands, num_nodes):
    assert (
        tours.shape[0] == demands.shape[0]
    ), "Batch size of tours and demands must match"
    assert isinstance(tours, np.ndarray) and isinstance(
        demands, np.ndarray
    ), "tours and demands must be numpy arrays"

    valid_tours = np.ones(tours.shape[0], dtype=bool)

    for i, tour in enumerate(tours):
        valid_tours[i] = _check_valid(tour, demands[i], num_nodes)

    return valid_tours


def beam_search(y_preds, batch_node_features, num_vehicles, beam_width=1280, **kwargs):
    y_preds = y_preds.cpu()
    batch_demands = (
        batch_node_features[..., 2].cpu() if batch_node_features is not None else None
    )

    y_preds = y_preds[..., 1]

    beamsearch = BeamSearch(
        y_preds,
        demands=batch_demands,
        beam_width=beam_width,
        num_vehicles=num_vehicles,
        **kwargs
    )
    beamsearch.search()

    return beamsearch


def get_tour_length_and_validity(
    current_tour, batch_dist_matrix, batch_demands, num_nodes
):
    tour_length = distance_from_adj_matrix(
        adj_matrix_from_routes(current_tour, num_nodes), batch_dist_matrix
    )
    valid_tour = check_valid_tours(current_tour, batch_demands, num_nodes)

    return current_tour, tour_length, valid_tour


def shortest_tour(
    y_preds,
    batch_dist_matrix,
    batch_node_features,
    num_vehicles,
    beam_width=1280,
    **kwargs
):
    bs = beam_search(y_preds, batch_node_features, num_vehicles, beam_width, **kwargs)

    batch_dist_matrix = batch_dist_matrix.cpu().numpy()
    batch_demands = batch_node_features[..., 2].cpu().numpy()

    shortest_tour = np.zeros((bs.batch_size, len(bs.next_nodes)), dtype=int)
    shortest_tour_length = np.full((bs.batch_size,), np.inf)

    results = Parallel(n_jobs=-1)(
        delayed(get_tour_length_and_validity)(
            bs.get_beam(i).numpy(), batch_dist_matrix, batch_demands, bs.num_nodes
        )
        for i in range(bs.beam_width)
    )

    for current_tour, tour_length, valid_tour in results:
        for i in range(bs.batch_size):
            if valid_tour[i] and tour_length[i] < shortest_tour_length[i]:
                shortest_tour[i] = current_tour[i]
                shortest_tour_length[i] = tour_length[i]

    return shortest_tour, shortest_tour_length


def most_probable_tour(
    y_preds,
    batch_dist_matrix,
    batch_node_features,
    num_vehicles,
    beam_width=1280,
    **kwargs
):
    bs = beam_search(y_preds, batch_node_features, num_vehicles, beam_width, **kwargs)

    # get most probable tours (index = 0)
    tours = bs.get_beam(0)

    tours = tours.cpu().numpy()
    batch_dist_matrix = batch_dist_matrix.cpu().numpy()

    tour_lengths = distance_from_adj_matrix(
        adj_matrix_from_routes(tours, batch_dist_matrix.shape[-1]), batch_dist_matrix
    )

    return tours, tour_lengths
