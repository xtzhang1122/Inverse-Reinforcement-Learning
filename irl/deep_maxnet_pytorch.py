import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from itertools import product

FLOAT = torch.float32

def find_svf(n_states, trajectories):
    """
    Find the state visitation frequency from trajectories.
    """
    svf = torch.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= len(trajectories)
    return svf

def optimal_value(n_states, n_actions, transition_probabilities, reward, discount, threshold=1e-2):
    """
    Find the optimal value function.
    """
    
    def until_converged():
        v = torch.zeros(n_states, dtype=FLOAT)
        while True:
            v_new = torch.zeros_like(v)
            for s in range(n_states):
                max_v = float("-inf")
                for a in range(n_actions):
                    tp = transition_probabilities[s, a, :]
                    max_v = max(max_v, tp @ (reward + discount * v))
                v_new[s] = max_v

            if torch.max(torch.abs(v - v_new)) < threshold:
                break
            v = v_new

        return v

    return until_converged()

def find_policy(n_states, n_actions, transition_probabilities, reward, discount, threshold=1e-2, v=None):
    """
    Find the optimal policy.
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward, discount, threshold)

    Q = torch.zeros((n_states, n_actions), dtype=FLOAT)
    for s in range(n_states):
        for a in range(n_actions):
            tp = transition_probabilities[s, a, :]
            Q[s, a] = tp @ (reward + discount * v)

    Q = torch.exp(Q - Q.max(dim=1, keepdim=True)[0])  # For numerical stability
    return Q / Q.sum(dim=1, keepdim=True)

def find_expected_svf(n_states, r, n_actions, discount, transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from Ziebart et al. 2008.
    """
    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    policy = find_policy(n_states, n_actions, transition_probability, r, discount)
    p_start_state = torch.bincount(trajectories[:, 0, 0], minlength=n_states).float() / n_trajectories
    svf = p_start_state

    for t in range(trajectory_length - 1):
        new_svf = torch.zeros(n_states)
        for s in range(n_states):
            for a in range(n_actions):
                new_svf += svf[s] * policy[s, a] * transition_probability[s, a, :]
        svf = new_svf

    return svf


def initialize_weights(model, init_type='normal', mean=0.0, std=0.02, a=-0.1, b=0.1):
    """
    Initializes weights of linear layers in the model.
    
    Parameters:
    - model: torch.nn.Module, the model containing the layers
    - init_type: str, 'normal' or 'uniform', type of initialization
    - mean: float, mean for normal distribution (only used if init_type is 'normal')
    - std: float, standard deviation for normal distribution (only used if init_type is 'normal')
    - a: float, lower bound for uniform distribution (only used if init_type is 'uniform')
    - b: float, upper bound for uniform distribution (only used if init_type is 'uniform')
    """
    for layer in model:
        if isinstance(layer, nn.Linear):
            if init_type == 'normal':
                init.normal_(layer.weight, mean=mean, std=std)
                init.normal_(layer.bias, mean=mean, std=std)
            elif init_type == 'uniform':
                init.uniform_(layer.weight, a=a, b=b)
                init.uniform_(layer.bias, a=a, b=b)
            else:
                raise ValueError("init_type must be 'normal' or 'uniform'")
            
    return model


def irl(structure, feature_matrix, n_actions, discount, transition_probability, trajectories, epochs, learning_rate, initialisation="normal", l1=0.1, l2=0.1):
    """
    Find the reward function for the given trajectories.
    """
    n_states, d_states = feature_matrix.shape
    transition_probability = torch.tensor(transition_probability, dtype=FLOAT)
    trajectories = torch.tensor(trajectories, dtype=torch.long)

    # Model initialization
    layers = []
    input_dim = structure[0]
    
    for hidden_dim in structure[1:]:
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        input_dim = hidden_dim
    layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers)
    model = initialize_weights(model, init_type='normal', mean=0.0, std=0.02)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        optimizer.zero_grad()
        r = model(feature_matrix).view(-1)
        r = (r - r.mean()) / r.std()

        expected_svf = find_expected_svf(n_states, r, n_actions, discount, transition_probability, trajectories)
        svf = find_svf(n_states, trajectories)

        loss = torch.sum((svf - expected_svf) ** 2) + l1 * torch.sum(torch.abs(r)) + l2 * torch.sum(r ** 2)
        loss.backward()
        optimizer.step()

    reward = model(feature_matrix).view(-1).detach()
    return reward.numpy()