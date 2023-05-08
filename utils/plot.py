import matplotlib.pyplot as plt


def plot_instance(instance, routes=None):
    """
    Plot the instance and the routes
    :param instance: (n, 3) instance of the problem
    :param dict routes: routes per vehicle to plot
    :return: None
    """
    plt.scatter(instance[1:, 0], instance[1:, 1])
    plt.scatter(instance[0, 0], instance[0, 1], c="r")

    if routes is not None:
        for route in routes.values():
            route = [0] + route + [0]
            locs = instance[route, :2]
            plt.plot(locs[:, 0], locs[:, 1])

    plt.show()
