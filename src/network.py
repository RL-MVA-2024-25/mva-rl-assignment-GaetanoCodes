"""network"""

import torch.nn as nn


def get_network(state_dim: int, nb_neurons: int, n_actions: int, device):
    # Utilisation de nn.Sequential pour empiler les couches
    return nn.Sequential(
        nn.Linear(state_dim, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, n_actions),
    ).to(device)
