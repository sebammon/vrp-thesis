import torch


def stable_topk(input_tensor, k, dim=-1, descending=True):
    # as a workaround for torch.topk(...) not being stable: https://github.com/pytorch/pytorch/issues/3982
    values, indices = torch.sort(input_tensor, dim=dim, descending=descending, stable=True)

    return values[..., :k], indices[..., :k]


class Beamsearch:
    """
    Beam search procedure class.

    References:
        [1]: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
        [2]: https://github.com/alexnowakvila/QAP_pt/blob/master/src/tsp/beam_search.py
        [3]: https://github.com/chaitjo/graph-convnet-tsp/blob/master/utils/beamsearch.py
    """

    def __init__(self, beam_width, trans_probs, num_vehicles=1, vehicle_capacity=1, random_start=False):
        # beam-search parameters
        self.beam_width = beam_width

        # TODO: Move tensors to GPU device for faster computation
        # tensor data types and device
        self.device = None
        self.float = torch.float32
        self.long = torch.int64

        # all transition probabilities
        self.trans_probs = trans_probs.type(self.float)
        self.batch_size = trans_probs.size(0)
        self.num_nodes = trans_probs.size(1)
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity

        assert len(trans_probs.shape) == 3, "transition probabilities need to be 3-dimensional"
        assert trans_probs.size(1) == trans_probs.size(2), "transition probabilities are not square"

        if random_start:
            # starting at random nodes
            start_nodes = torch.randint(0, self.num_nodes, (self.batch_size, self.beam_width))
        else:
            # starting at node zero
            start_nodes = torch.zeros(self.batch_size, self.beam_width)

        self.start_nodes = start_nodes.type(self.long)
        self.depot_visits_counter = torch.zeros(self.batch_size, self.beam_width)
        self.remaining_capacity = torch.ones(self.batch_size, self.beam_width) * self.vehicle_capacity

        # mask for removing visited nodes etc.
        self.mask = torch.ones(self.batch_size, self.beam_width, self.num_nodes).type(self.float)

        # transition probability scores up-until current timestep
        self.scores = torch.zeros(self.batch_size, self.beam_width).type(self.float)

        # pointers to parents for each timestep
        self.parent_pointer = []

        # nodes at each timestep
        self.next_nodes = [self.start_nodes]

        # start by masking the starting nodes
        self.update_mask(self.start_nodes)

    def get_current_nodes(self):
        """
        Get the nodes to expand at the current timestep
        """
        current_nodes = self.next_nodes[-1]
        current_nodes = current_nodes.unsqueeze(2).expand_as(self.mask)

        return current_nodes

    @property
    def num_iterations(self):
        # -1 for num_nodes because we already start at depot
        # -1 to offset num_vehicles
        return self.num_nodes + self.num_vehicles - 2

    def search(self):
        """
        Start beam search
        """
        for step in range(self.num_iterations):
            self.step()

    def step(self):
        """
        Transition to the next timestep of the beam search
        """
        current_nodes = self.get_current_nodes()
        trans_probs = self.trans_probs.gather(1, current_nodes)

        if len(self.parent_pointer) == 0:
            # first transition, only use the starting nodes
            beam_prob = trans_probs
            beam_prob[:, 1:] = torch.zeros_like(beam_prob[:, 1:])
        else:
            # multiply the previous scores (probabilities) with the current ones
            expanded_scores = self.scores.unsqueeze(2).expand_as(trans_probs)  # b x beam_width x num_nodes
            beam_prob = trans_probs * expanded_scores

        # mask out visited nodes
        beam_prob = beam_prob * self.mask
        beam_prob[..., 0] += 1e-25  # always make the depot slightly available

        beam_prob = beam_prob.view(beam_prob.size(0), -1)  # flatten to (b x beam_width * num_nodes)

        # get k=beam_width best scores and indices
        best_scores, best_score_idxs = stable_topk(beam_prob, k=self.beam_width, dim=1)

        self.scores = best_scores
        parent_index = torch.floor_divide(best_score_idxs, self.num_nodes).type(self.long)
        self.parent_pointer.append(parent_index)

        # next nodes
        next_node = best_score_idxs - (parent_index * self.num_nodes)  # convert flat indices back to original
        self.next_nodes.append(next_node)

        # keep masked rows from parents (for next step)
        parent_mask = parent_index.unsqueeze(2).expand_as(self.mask)  # (batch_size, beam_size, num_nodes)
        self.mask = self.mask.gather(1, parent_mask)

        # keep depot counter and capacity from parent (for next step)
        self.depot_visits_counter = self.depot_visits_counter.gather(1, parent_index)
        self.remaining_capacity = self.remaining_capacity.gather(1, parent_index)

        # mask next nodes (newly added nodes)
        self.update_mask(next_node)

    def update_mask(self, new_nodes):
        """
        Sets indices of new_nodes = 0 in the mask.

        Args:
            new_nodes: (batch_size, beam_width) of new node indices
        """
        index = torch.arange(0, self.num_nodes, dtype=self.long).expand_as(self.mask)
        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask)

        # set the mask = 0 at the new_node_idx positions
        visited_nodes_mask = torch.eq(index, new_nodes).type(self.float)
        update_mask = 1 - visited_nodes_mask

        # increment depot visit counter where visited
        self.depot_visits_counter += visited_nodes_mask[..., 0]  # batch_size x beam_width x num_nodes[0]
        enable_depot_visit = torch.lt(self.depot_visits_counter, self.num_vehicles).type(self.long)
        # enable_depot_visit = enable_depot_visit * update_mask[..., 0]

        # TODO:
        # 1. reset the remaining capacity when visiting depot
        # 2. decrement remaining capacity when visiting new nodes
        # 3. mask nodes that don't fit in remaining capacity

        self.mask = self.mask * update_mask

        # reset depot visit
        self.mask[..., 0] = enable_depot_visit

    def get_beam(self, beam_idx):
        """
        Construct the beam for the given index

        Args:
            beam_idx int: Index of the beam to construct (0 = best, ..., n = worst)
        """
        assert len(self.next_nodes) == self.num_iterations + 1

        prev_pointer = torch.ones(self.batch_size, 1).type(self.long) * beam_idx
        last_node = self.next_nodes[-1].gather(1, prev_pointer)

        path = [last_node]

        for i in range(len(self.parent_pointer) - 1, -1, -1):
            prev_pointer = self.parent_pointer[i].gather(1, prev_pointer)
            last_node = self.next_nodes[i].gather(1, prev_pointer)

            path.append(last_node)

        path = list(reversed(path))
        path = torch.cat(path, dim=-1)

        return path

    def validate(self, beam, batch_idx=None, beam_idx=None):
        bin_count = torch.bincount(beam)

        assert bin_count[
                   0] <= self.num_vehicles, f"Batch={batch_idx}, beam={beam_idx}: too many depot visits {bin_count[0]} > {self.num_vehicles}\n{beam}"
        # want them separate for sanity
        assert torch.all(bin_count[1:] <= 1), f"Batch={batch_idx}, beam={beam_idx}: too many node visits\n{beam}"
        assert torch.all(bin_count[1:] > 0), f"Batch={batch_idx}, beam={beam_idx}: not all nodes visited\n{beam}"

    def sanity_check(self):
        for batch_idx in range(self.batch_size):
            for beam_idx in range(self.beam_width):
                beams = self.get_beam(beam_idx)
                beam = beams[batch_idx]

                self.validate(beam, batch_idx=batch_idx, beam_idx=beam_idx)
