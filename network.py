import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, activation_functions):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.layers.append(self.add_activation(activation_functions[0]))
        i = 1
        while i < len(hidden_layers):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.layers.append(self.add_activation(activation_functions[i]))
            i += 1
        self.layers.append(nn.Linear(hidden_layers[i - 1], num_classes))
        self.model_architect = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model_architect(x)

    def add_activation(self, activation_function):
        if activation_function == 'ReLu':
            return nn.ReLU()
        elif activation_function == 'Tanh':
            return nn.Tanh()
        elif activation_function == 'Sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()
