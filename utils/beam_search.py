import torch


def stable_topk(input_tensor, k, dim=-1, descending=True):
    # as a workaround for torch.topK(...) not being stable: https://github.com/pytorch/pytorch/issues/3982
    values, indices = torch.sort(
        input_tensor, dim=dim, descending=descending, stable=True
    )

    return values[..., :k], indices[..., :k]


class BeamSearch:
    """
    Beam search procedure class.

    References:
        [1]: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
        [2]: https://github.com/alexnowakvila/QAP_pt/blob/master/src/tsp/beam_search.py
        [3]: https://github.com/chaitjo/graph-convnet-tsp/blob/master/utils/beamsearch.py
    """

    def __init__(
            self,
            trans_probs,
            num_vehicles,
            beam_width=1,
            demands=None,
            vehicle_capacity=1,
            random_start=False,
            allow_consecutive_visits=False,
    ):
        # beam-search parameters
        self.beam_width = beam_width
        self.allow_consecutive_visits = allow_consecutive_visits

        self.device = trans_probs.device
        self.float = torch.float32
        self.long = torch.int64

        assert isinstance(
            trans_probs, torch.Tensor
        ), "transition probabilities need to be a tensor"
        assert (
                len(trans_probs.shape) == 3
        ), "transition probabilities need to be 3-dimensional"
        assert trans_probs.size(1) == trans_probs.size(
            2
        ), "transition probabilities are not square"

        if demands is not None:
            assert isinstance(demands, torch.Tensor), "demands need to be a tensor"
            assert len(demands.shape) == 2, "demands need to be 2-dimensional"

        # all transition probabilities
        self.trans_probs = trans_probs.type(self.float)
        self.demands = demands.to(self.device) if demands is not None else None
        self.batch_size = trans_probs.size(0)
        self.num_nodes = trans_probs.size(1)
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity

        if random_start:
            # starting at random nodes
            start_nodes = torch.randint(
                0, self.num_nodes, (self.batch_size, self.beam_width)
            )
        else:
            # starting at node zero
            start_nodes = torch.zeros(self.batch_size, self.beam_width)

        self.start_nodes = start_nodes.type(self.long).to(self.device)
        self.depot_visits_counter = torch.zeros(
            self.batch_size, self.beam_width, device=self.device
        )
        self.remaining_capacity = (
                torch.ones(self.batch_size, self.beam_width, device=self.device)
                * self.vehicle_capacity
        )

        # mask for removing visited nodes etc.
        self.unvisited_mask = torch.ones(
            self.batch_size,
            self.beam_width,
            self.num_nodes,
            dtype=self.long,
            device=self.device,
        )

        # transition probability scores up-until current timestep
        self.scores = torch.zeros(
            self.batch_size, self.beam_width, dtype=self.float, device=self.device
        )

        # pointers to parents for each timestep
        self.parent_pointer = []

        # nodes at each timestep
        self.next_nodes = [self.start_nodes]

        # start by masking the starting nodes
        self.update_mask(self.start_nodes)

    @property
    def current_nodes(self):
        """
        Get the nodes to expand at the current timestep
        """
        current_nodes = self.next_nodes[-1]
        current_nodes = current_nodes.unsqueeze(2).expand_as(self.unvisited_mask)

        return current_nodes

    @property
    def num_iterations(self):
        # -1 for num_nodes because we already start at depot
        # -1 to offset num_vehicles
        return self.num_nodes + self.num_vehicles - 2

    @property
    def capacity_mask(self):
        demand = self.demands.unsqueeze(1)
        capacity = self.remaining_capacity.unsqueeze(2)

        # batch_size x beam_width x num_nodes (same as visited_mask)
        mask = torch.le(demand, capacity).type(self.long)
        mask[..., 0] = 1  # depot always available

        return mask

    @property
    def mask(self):
        mask = self.unvisited_mask.clone()

        if self.demands is not None:
            remaining_nodes_available = self.capacity_mask.sum(-1)
            # only depot available
            mask[..., 0] = torch.eq(remaining_nodes_available, 1).type(self.long)
            mask *= self.capacity_mask

        elif len(self.next_nodes) > 1 and not self.allow_consecutive_visits:
            # mask out depot node that was just visited
            mask_depot_node = 1 - torch.eq(self.next_nodes[-2], 0).type(self.float)
            mask[..., 0] = mask[..., 0] * mask_depot_node

        return mask

    def search(self):
        """
        Start beam search
        """
        if self.num_vehicles > 0:
            for step in range(self.num_iterations):
                self.step()
        else:
            while self.unvisited_mask.sum() > 0:
                self.step()

    def step(self):
        """
        Transition to the next timestep of the beam search
        """
        current_nodes = self.current_nodes
        trans_probs = self.trans_probs.gather(1, current_nodes)

        if len(self.parent_pointer) == 0:
            # first transition
            beam_prob = trans_probs
            # use only the starting nodes
            beam_prob[:, 1:] = torch.zeros_like(beam_prob[:, 1:])
        else:
            # multiply the previous scores (probabilities) with the current ones
            expanded_scores = self.scores.unsqueeze(2).expand_as(
                trans_probs
            )  # b x beam_width x num_nodes
            beam_prob = trans_probs * expanded_scores

        # mask out nodes (based on conditions)
        beam_prob = beam_prob * self.mask
        beam_prob += 1e-25  # avoid creating zero probability tours

        beam_prob = beam_prob.view(
            beam_prob.size(0), -1
        )  # flatten to (b x beam_width * num_nodes)

        # get k=beam_width best scores and indices (stable)
        best_scores, best_score_idx = stable_topk(beam_prob, k=self.beam_width, dim=1)

        self.scores = best_scores
        parent_index = torch.floor_divide(best_score_idx, self.num_nodes).type(
            self.long
        )
        self.parent_pointer.append(parent_index)

        # next nodes
        next_node = best_score_idx - (
                parent_index * self.num_nodes
        )  # convert flat indices back to original
        self.next_nodes.append(next_node)

        # keep masked rows from parents (for next step)
        parent_mask = parent_index.unsqueeze(2).expand_as(
            self.unvisited_mask
        )  # batch_size x beam_size x num_nodes
        self.unvisited_mask = self.unvisited_mask.gather(1, parent_mask)

        # keep depot counter and capacity from parent (for next step)
        self.depot_visits_counter = self.depot_visits_counter.gather(1, parent_index)
        self.remaining_capacity = self.remaining_capacity.gather(1, parent_index)

        # mask next nodes (newly added nodes)
        self.update_mask(next_node)

    def update_mask(self, nodes):
        """
        Updates mask by setting visited nodes = 0.

        :param nodes: (batch_size, beam_width) of new node indices
        """
        index = torch.arange(
            0, self.num_nodes, dtype=self.long, device=self.device
        ).expand_as(self.unvisited_mask)
        new_nodes = nodes.unsqueeze(2).expand_as(self.unvisited_mask)

        visited_nodes_mask = torch.eq(index, new_nodes).type(self.long)

        # set the mask = 0 at the new_node_idx positions
        unvisited_update_mask = 1 - visited_nodes_mask

        # increment depot counter when visited
        self.depot_visits_counter += visited_nodes_mask[
            ..., 0
        ]  # batch_size x beam_width x num_nodes[0]
        enable_depot_visit = torch.lt(
            self.depot_visits_counter, self.num_vehicles
        ).type(self.long)
        enable_depot_visit *= unvisited_update_mask[..., 0]

        # set new mask
        self.unvisited_mask *= unvisited_update_mask

        # reset depot visit
        if self.demands is not None:
            # decrement remaining capacity
            loads = self.demands.gather(1, nodes)
            self.remaining_capacity -= loads

            # reset depot visits, otherwise keep
            self.remaining_capacity = torch.maximum(
                self.vehicle_capacity * visited_nodes_mask[..., 0],
                self.remaining_capacity,
            )
        else:
            self.unvisited_mask[..., 0] = enable_depot_visit

    def get_beam(self, beam_idx):
        """
        Construct the beam for the given index

        :param int beam_idx: Index of the beam to construct (0 = best, ..., n = worst)
        """
        if self.num_vehicles > 0:
            assert len(self.next_nodes) == self.num_iterations + 1

        paths = (
            torch.ones(self.batch_size, len(self.next_nodes))
            .type(self.long)
            .to(self.device)
        )
        prev_pointer = (
                torch.ones(self.batch_size, 1).type(self.long).to(self.device) * beam_idx
        )
        last_node = self.next_nodes[-1].gather(1, prev_pointer)

        paths[:, -1] = last_node.view(1, self.batch_size)

        for i in range(len(self.parent_pointer) - 1, -1, -1):
            prev_pointer = self.parent_pointer[i].gather(1, prev_pointer)
            last_node = self.next_nodes[i].gather(1, prev_pointer)

            paths[:, i] = last_node.view(1, self.batch_size)

        return paths
