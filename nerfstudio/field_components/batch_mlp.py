# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Batch Multi Layer Perceptron
"""
import math
from typing import Optional, Set, Tuple

import torch
from torch import Tensor, nn
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.utils.cpp_extension import load
from torchtyping import TensorType

from nerfstudio.field_components.base_field_component import FieldComponent


class BatchMLP(FieldComponent):
    """Batch Multilayer perceptron. Each sample in a batch has its own weight matrix.
        Since the batch size is not fixed, the weight matrix has to be given along with
        the input.

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each BatchMLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        residual_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
    ) -> None:

        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self._residual_connections: Set[int] = set(residual_connections) if residual_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize batch multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(BatchLinear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    assert (
                        i not in self._residual_connections
                    ), "Residual connection at layer 0 could cause shape mismatch."
                    layers.append(BatchLinear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(BatchLinear(self.layer_width + self.in_dim, self.layer_width))
                else:
                    layers.append(BatchLinear(self.layer_width, self.layer_width))
            layers.append(BatchLinear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(
        self, in_tensor: TensorType["bs":..., "in_dim"], weight_tensor: TensorType["bs":..., "weight_dim"]
    ) -> TensorType["bs":..., "out_dim"]:
        """Process input with a batch multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            BatchMLP network output
        """
        in_tensor = in_tensor.unsqueeze(1)

        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)

            x_last = x
            x = layer(x, weight_tensor[:, i])

            if i in self._residual_connections:
                x = x + x_last

            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)

        if self.out_activation is not None:
            x = self.out_activation(x)
        return x.squeeze(1)


class BatchLinear(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias

        # self.weight = Parameter(torch.empty((num_group, out_dim, in_dim), **factory_kwargs))
        # self.bias = Parameter(torch.empty(num_group, out_dim, **factory_kwargs))

    def forward(self, input: Tensor, weight: Tensor) -> Tensor:
        num_param_w = self.in_dim * self.out_dim
        assert weight.shape[1] >= num_param_w
        w = weight[:, :num_param_w].reshape(weight.shape[0], self.out_dim, self.in_dim)
        if self.bias:
            assert weight.shape[1] >= num_param_w + self.out_dim
            b = weight[:, num_param_w : num_param_w + self.out_dim].unsqueeze(1)
        else:
            b = None
        return BatchLinearFunction.apply(input, w, b)

    def extra_repr(self) -> str:
        return "in_dim={}, out_dim={}, bias={}, dtype={}".format(
            self.in_dim, self.out_dim, self.bias is not None, self.weight.dtype
        )


class BatchLinearFunction(torch.autograd.Function):
    """BatchLinear <Python>"""

    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.bmm(weight.transpose(-1, -2))  # use `bmm` instead of `mm` to parallelize along BatchMLPs

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.bmm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(1, 2).bmm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output
        return grad_input, grad_weight, grad_bias


if __name__ == "__main__":
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.

    input = (
        torch.randn(2, 10, 3, dtype=torch.double, requires_grad=True),
        torch.randn(2, 64, 3, dtype=torch.double, requires_grad=True),
        torch.randn(2, 64, dtype=torch.double, requires_grad=True),
    )
    test = gradcheck(BatchLinearFunction.apply, input, eps=1e-6, atol=1e-4)
    print("BatchLinear", test)
