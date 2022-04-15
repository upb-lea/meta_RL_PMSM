# Actor and Critic code altered from https://github.com/sfujim/TD3
# Context code altered from https://github.com/amazon-research/meta-q-learning
import torch.nn as nn
import torch

def create_FF_network(hidden_layers, input_dim, output_dim):
    # Creates list of feed forward layers based on dimensions presented
    layers = nn.ModuleList()
    layers.append(nn.Linear(input_dim, hidden_layers[0]))
    layers.append(nn.ReLU())
    for i in range(len(hidden_layers) - 1):
        layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_layers[-1], output_dim))
    return layers


class Actor(nn.Module):
    def __init__(self, hidden_layers, state_dim, action_dim, max_action, context_size=0):
        #hidden layers is a list of integers which represent the amount of units in the hidden layers
        #e.g. [256,256]
        super(Actor, self).__init__()
        self.layers = create_FF_network(hidden_layers, input_dim=state_dim+context_size, output_dim=action_dim)
        self.max_action = max_action

    def forward(self, state,context=None):
        if context is not None:
            out = torch.cat([state,context], dim=1)
        else:
            out = state
        for layer in self.layers:
            out = layer(out)
        out = self.max_action * torch.tanh(out)

        return out


class Critic(nn.Module):
    def __init__(self, hidden_layers, state_dim, action_dim, context_size=0):
        #hidden layers is a list of integers which represent the amount of units in the hidden layers
        #e.g. [256,256]
        super(Critic, self).__init__()
        self.layers_q1 = create_FF_network(hidden_layers, input_dim=state_dim+action_dim+context_size, output_dim=1)
        self.layers_q2 = create_FF_network(hidden_layers, input_dim=state_dim+action_dim+context_size, output_dim=1)


    def forward(self, state, action, context=None, which='both'):
        out_q1 = self.Q_val(state,action,1, context)
        if which == 'Q1':
            return out_q1
        elif which == 'both':
            out_q2 = self.Q_val(state,action,2, context)
            return out_q1, out_q2

    def Q_val(self, state, action, net, context=None):
        if net == 1:
            layers = self.layers_q1
        elif net == 2:
            layers = self.layers_q2
        if context is not None:
            out = torch.cat([state,action,context], 1)
        else:
            out = torch.cat([state,action], 1)
        for layer in layers:
            out = layer(out)
        return torch.tanh(out)

class Context(nn.Module):
    def __init__(self,
                 hidden_layers,
                 output_dim=None,
                 input_dim=None):

        super(Context, self).__init__()
        self.hidden_sizes = output_dim
        self.input_dim = input_dim
        self.layers = create_FF_network(hidden_layers,input_dim, output_dim)

    def forward(self, transitions):
        out = transitions
        for layer in self.layers:
            out = layer(out)
        out = torch.mean(out, dim=1)
        return torch.tanh(out)