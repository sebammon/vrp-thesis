from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


class Solver:
    def __init__(
        self,
        data,
        time_limit=5,
        first_solution_strategy="SAVINGS",
        local_search_metaheuristic="GUIDED_LOCAL_SEARCH",
    ):
        """
        :type data: Instance
        :param dict data: data of the problem
        :param int time_limit: maximum time to run the solver
        :param str first_solution_strategy: strategy to use to find the first solution
        :param str local_search_metaheuristic: metaheuristic to use for local search
        """
        self.data = data
        self.solution = None
        self.manager = None
        self.routing = None

        # configs
        self.time_limit = time_limit
        self.first_solution_strategy = getattr(
            routing_enums_pb2.FirstSolutionStrategy, first_solution_strategy
        )
        self.local_search_metaheuristic = getattr(
            routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristic
        )

    def distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.data["distance_matrix"][from_node][to_node]

    def demand_callback(self, from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        return self.data["demands"][from_node]

    def solve(self):
        data = self.data

        # Create the routing index manager.
        self.manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        self.routing = pywrapcp.RoutingModel(self.manager)

        # Create and register a transit callback.
        transit_callback_index = self.routing.RegisterTransitCallback(
            self.distance_callback
        )

        # Define cost of each arc.
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(
            self.demand_callback
        )
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data["vehicle_capacities"],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = self.first_solution_strategy
        search_parameters.local_search_metaheuristic = self.local_search_metaheuristic
        search_parameters.time_limit.FromSeconds(self.time_limit)

        # Solve the problem.
        self.solution = self.routing.SolveWithParameters(search_parameters)

        return self.solution

    def get_routes(self):
        if self.solution is None:
            return []

        routes = []
        for route_nbr in range(self.routing.vehicles()):
            index = self.routing.Start(route_nbr)
            route = [self.manager.IndexToNode(index)]
            while not self.routing.IsEnd(index):
                index = self.solution.Value(self.routing.NextVar(index))
                route.append(self.manager.IndexToNode(index))

            # ignore empty routes
            if len(route) > 2:
                routes.append(route)

        return routes

    def print_solution(self):
        print(f"Objective: {self.solution.ObjectiveValue()}")
        total_distance = 0
        total_load = 0
        for vehicle_id in range(self.data["num_vehicles"]):
            index = self.routing.Start(vehicle_id)
            plan_output = "Route for vehicle {}:\n".format(vehicle_id)
            route_distance = 0
            route_load = 0
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                route_load += self.data["demands"][node_index]
                plan_output += " {0} Load({1}) -> ".format(node_index, route_load)
                previous_index = index
                index = self.solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            plan_output += " {0} Load({1})\n".format(
                self.manager.IndexToNode(index), route_load
            )
            plan_output += "Distance of the route: {}m\n".format(route_distance)
            plan_output += "Load of the route: {}\n".format(route_load)
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print("Total distance of all routes: {}m".format(total_distance))
        print("Total load of all routes: {}".format(total_load))
