"""
PyTorch autograd functions.
NOTE: We observe that PyTorch's autograd engine can efficiently handle certain operations without
requiring manual `torch.autograd.Function` implementations. These operations have been refactored
into standard Python functions while maintaining backward-compatible interfaces.
"""

from abc import ABC, abstractmethod
import torch
from .utils.typing import Tuple

IndicesTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class FunctionABC(ABC):
    @staticmethod
    @abstractmethod
    def apply(*args, **kwargs):
        pass


class UngroupFunction(FunctionABC):
    @staticmethod
    def apply(
        x: torch.Tensor,  # NOTE: Shape [*seqs, *others]
        indices: IndicesTuple,  # 4 non-zero mask index tensors
        shapes: Tuple[
            torch.Size, torch.Size
        ],  # shapes of ungrouped prefix and ungrouped suffix
    ):
        input_shape = x.shape
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = indices
        (
            prefix_x_shape,
            suffix_x_shape,
        ) = shapes
        other_shapes = input_shape[len(prefix_x_shape):]
        # Split the grouped inputs into prefix and suffix tensors.
        prefix_x = torch.zeros(
            *prefix_x_shape,
            *other_shapes,
            dtype=x.dtype,
            device=x.device,
        )
        suffix_x = torch.zeros(
            *suffix_x_shape,
            *other_shapes,
            dtype=x.dtype,
            device=x.device,
        )
        prefix_x[tuple(ungrouped_prefix_indices.T)] = x[tuple(grouped_prefix_indices.T)]
        suffix_x[tuple(ungrouped_suffix_indices.T)] = x[tuple(grouped_suffix_indices.T)]
        return prefix_x, suffix_x


class GroupFunction(FunctionABC):
    @staticmethod
    def apply(
        prefix_x: torch.Tensor,  # NOTE: Shape [*seqs, *others]
        suffix_x: torch.Tensor,  # NOTE: Shape [*seqs, *others]
        indices: IndicesTuple,  # 4 non-zero mask index tensors
        x_shape: torch.Size,  # shape of grouped input x
    ):
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = indices
        other_shapes = prefix_x.shape[len(x_shape):]
        # Concat the prefix and suffix inputs into a single grouped input tensor
        x = torch.zeros(
            *x_shape,
            *other_shapes,
            dtype=prefix_x.dtype,
            device=prefix_x.device,
        )
        x[tuple(grouped_prefix_indices.T)] = prefix_x[tuple(ungrouped_prefix_indices.T)]
        x[tuple(grouped_suffix_indices.T)] = suffix_x[tuple(ungrouped_suffix_indices.T)]
        return x


class ConvertPaddingFunction(FunctionABC):
    @staticmethod
    def apply(
        x: torch.Tensor,  # NOTE: Shape: [*seqs, *others]
        indices: Tuple[torch.Tensor, torch.Tensor],  # 2 non-zero mask index tensors
        o_shape: torch.Size,  # shape of converted output tensor at seq dims
    ):
        input_shape = x.shape
        x_indices, o_indices = indices
        other_shapes = input_shape[len(o_shape):]
        o = torch.zeros(
            *o_shape,
            *other_shapes,
            dtype=x.dtype,
            device=x.device,
        )
        o[tuple(o_indices.T)] = x[tuple(x_indices.T)]
        return o
