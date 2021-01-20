from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LayerSize = int


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        fc_layers: Iterable[LayerSize],
        final_activation: Optional[Callable],
    ):
        super().__init__()
        self.final_activation = final_activation
        layer_sizes = [input_size] + list(fc_layers) + [output_size]
        self.fc_layers = []

        for i, (layer_input_size, layer_output_size) in enumerate(
            zip(layer_sizes, layer_sizes[1:])
        ):
            layer = nn.Linear(layer_input_size, layer_output_size)
            setattr(self, f"fc{i}", layer)
            self.fc_layers.append(layer)

    @property
    def last_layer(self) -> nn.Linear:
        return self.fc_layers[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.fc_layers[:-1]:
            x = F.leaky_relu(layer(x))
        if self.final_activation:
            return self.final_activation(self.last_layer(x))
        else:
            return self.last_layer(x)


class Actor(NeuralNetwork):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        fc_layers: Iterable[LayerSize] = (64, 64),
    ):
        super().__init__(state_size, action_size, fc_layers, F.tanh)


class Critic(NeuralNetwork):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        fc_layers: Iterable[LayerSize] = (64, 64),
    ):
        super().__init__(state_size + action_size, 1, fc_layers, None)

    def __call__(self, states: torch.Tensor, actions: torch.Tensor):
        x = torch.cat((states, actions), dim=1)
        return super().__call__(x)
