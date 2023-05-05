# GOOGLE OR TOOLS SOLVER - from the Google OR Tools documentation
# https://developers.google.com/optimization/routing/cvrp

import argparse
import pickle
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import tqdm
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

SCALING_FACTOR = 1_000


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def create_data_model(locations, demands, vehicle_capacities, num_vehicles, depot_index=0):
    """Stores the data for the problem."""
    data_model = {}

    locations = np.array(
        locations) * SCALING_FACTOR  # Google OR Tools requires integer values, so we scale the locations
    distance_matrix = np.zeros((len(locations), len(locations)))

    for i in range(len(locations)):
        for j in range(i, len(locations)):
            distance_matrix[i][j] = distance(locations[i], locations[j])
            distance_matrix[j][i] = distance_matrix[i][j]

    # Note the conversions to int and list
    data_model['distance_matrix'] = np.round(distance_matrix, 0).astype(int).tolist()
    data_model['demands'] = demands.astype(int).tolist()
    data_model['vehicle_capacities'] = vehicle_capacities.astype(int).tolist()
    data_model['num_vehicles'] = num_vehicles
    data_model['depot'] = depot_index

    return data_model


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))


def get_solution(data, manager, routing, solution):
    solution_data = {}
    classifications = np.full(len(data['distance_matrix']), -1)
    routes = {}

    for vehicle_id in range(data['num_vehicles']):
        route = []
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            index = solution.Value(routing.NextVar(index))

            if node_index != data['depot']:
                classifications[node_index] = vehicle_id
                route.append(node_index)

        routes[vehicle_id] = route

    classifications[data['depot']] = -1

    solution_data['classifications'] = classifications
    solution_data['routes'] = routes
    solution_data['total_distance'] = solution.ObjectiveValue() / SCALING_FACTOR

    return solution_data


def solve(data):
    """Solve the CVRP problem."""

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(5)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        # print_solution(data, manager, routing, solution)
        return get_solution(data, manager, routing, solution)
    else:
        return {}


def task(i, instance, num_vehicles, vehicle_capacity):
    locations = instance[:, :2]
    demands = instance[:, 2]

    vehicle_capacities = np.full(num_vehicles, vehicle_capacity)

    data = create_data_model(locations=locations, demands=demands, vehicle_capacities=vehicle_capacities,
                             num_vehicles=num_vehicles)

    solution = solve(data)
    solution['instance'] = i

    return solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--num_vehicles', type=int, required=True)
    parser.add_argument('--vehicle_capacity', type=int, required=True)

    args = parser.parse_args()

    filename = args.filename
    num_vehicles = args.num_vehicles
    vehicle_capacity = args.vehicle_capacity

    RESULTS_PATH = Path("./results")

    if not RESULTS_PATH.exists():
        RESULTS_PATH.mkdir()

    with open(filename, 'rb') as f:
        instances = pickle.load(f)

    task_args = [(i, instance, num_vehicles, vehicle_capacity) for i, instance in enumerate(instances)]

    start_time = time.time()
    count = cpu_count()

    print("Starting processing with {} workers".format(count))

    with Pool(count) as p:
        # TODO: tqdm doesn't work with starmap
        results = p.starmap(task, tqdm.tqdm(task_args, total=len(task_args)))

    with open(RESULTS_PATH / f'vrp_{num_vehicles}_{vehicle_capacity}.pkl', 'wb') as f:
        pickle.dump(results, f)

    end_time = time.time()

    print("Done in {} seconds".format(end_time - start_time))
