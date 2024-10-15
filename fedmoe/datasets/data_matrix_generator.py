from abc import ABC, abstractmethod
from typing import List

import torch

from fedmoe.utils.typing import InputGenerationFunction, TargetGenerationFunction


class InputGenerator(ABC):
    @abstractmethod
    def generate_input_tensor(self, time_axis: torch.Tensor) -> torch.Tensor:
        pass


class TargetGenerator(ABC):
    @abstractmethod
    def generate_target_tensor(self, time_axis: torch.Tensor, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        Implementation of how to generate output at each time step which can depend on input as well.
        Ideally, we want to have a relation between input and output sequences,
        to make sure output is relevant to input, but this is not enforced.
        """
        pass


class MultiDimensionalTimeFunctionInputGenerator(InputGenerator):
    def __init__(self, function_list: List[InputGenerationFunction], x_dim: int):
        self.function_list = function_list
        self.x_dim = x_dim
        assert len(self.function_list) == x_dim

    def generate_input_tensor(self, time_axis: torch.Tensor) -> torch.Tensor:
        input_dimension_list = []
        for func_x in self.function_list:
            input_dimension_list.append(func_x(time_axis))

        input_matrix = torch.stack(input_dimension_list, dim=1).double()
        # Input matrix's shape should be (time_steps, x_dim)
        assert input_matrix.shape == (
            time_axis.shape[0],
            self.x_dim,
        ), f"Error: input matrix's shape is {input_matrix.shape}"
        return input_matrix


class MultiDimensionalTargetGenerator(TargetGenerator):
    def __init__(
        self,
        function_list: List[TargetGenerationFunction],
        y_dim: int,
    ):
        self.function_list = function_list
        # We need to have a function for each target dimension.
        assert len(self.function_list) == y_dim
        self.y_dim = y_dim

    def generate_target_tensor(self, time_axis: torch.Tensor, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        Generates target tensor that could be a function of time and input matrix.
        Each dimension of target (yn) has a 1 to D relation to all the columns in the
        input matrix (D is the input dimension).
        """
        y_list = []
        for y_idx in range(0, self.y_dim):
            # last input to the function is always time
            y_list.append(self.function_list[y_idx](input_matrix, time_axis))

        target_matrix = torch.stack(y_list, dim=1).double()

        # Target matrix's shape should be (time_steps, y_dim)
        assert target_matrix.shape == (
            time_axis.shape[0],
            self.y_dim,
        ), f"Error: input matrix's shape is {target_matrix.shape}"
        return target_matrix
