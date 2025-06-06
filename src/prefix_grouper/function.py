"""
PyTorch autograd functions.
"""

import torch
from torch.autograd import Function
from .utils.typing import Tuple

IndicesTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class UngroupFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,  # NOTE: Shape: [b, num_heads (or 1 for common tensors), seq, ...]
        indices: IndicesTuple,  # 4 non-zero mask index tensors
        shapes: Tuple[
            torch.Size, torch.Size
        ],  # shapes of ungrouped prefix and ungrouped suffix
    ):
        # NOTE: This function can accept [b, num_heads (or 1 for common tensors), seq, ...]
        assert x.ndim >= 3
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
        ctx.save_for_backward(*indices)
        ctx.input_shape = input_shape
        # Split the grouped inputs into prefix and suffix tensors.
        prefix_x = torch.zeros(
            prefix_x_shape[0],
            input_shape[1],
            prefix_x_shape[1],
            *input_shape[3:],
            dtype=x.dtype,
            device=x.device,
        )
        suffix_x = torch.zeros(
            suffix_x_shape[0],
            input_shape[1],
            suffix_x_shape[1],
            *input_shape[3:],
            dtype=x.dtype,
            device=x.device,
        )
        prefix_x[ungrouped_prefix_indices[:, 0], :, ungrouped_prefix_indices[:, 1]] = x[
            grouped_prefix_indices[:, 0], :, grouped_prefix_indices[:, 1]
        ]
        suffix_x[ungrouped_suffix_indices[:, 0], :, ungrouped_suffix_indices[:, 1]] = x[
            grouped_suffix_indices[:, 0], :, grouped_suffix_indices[:, 1]
        ]
        return prefix_x, suffix_x

    @staticmethod
    def backward(ctx, grad_prefix_x: torch.Tensor, grad_suffix_x: torch.Tensor):
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = ctx.saved_tensors
        input_shape = ctx.input_shape
        # Concat the prefix and suffix grad into a single tensor.
        grad_x = torch.zeros(
            input_shape, dtype=grad_prefix_x.dtype, device=grad_prefix_x.device
        )
        grad_x[grouped_prefix_indices[:, 0], :, grouped_prefix_indices[:, 1]] = (
            grad_prefix_x[
                ungrouped_prefix_indices[:, 0], :, ungrouped_prefix_indices[:, 1]
            ]
        )
        grad_x[grouped_suffix_indices[:, 0], :, grouped_suffix_indices[:, 1]] = (
            grad_suffix_x[
                ungrouped_suffix_indices[:, 0], :, ungrouped_suffix_indices[:, 1]
            ]
        )
        return grad_x, None, None


class GroupFunction(Function):
    @staticmethod
    def forward(
        ctx,
        prefix_x: torch.Tensor,  # NOTE: Shape [b, seq, ...]
        suffix_x: torch.Tensor,  # NOTE: Shape [b, seq, ...]
        indices: IndicesTuple,  # 4 non-zero mask index tensors
        x_shape: torch.Size,  # shape of grouped input x
    ):
        # NOTE: This function can accept [b, seq, ...]
        assert prefix_x.ndim == suffix_x.ndim >= 2
        prefix_shape, suffix_shape = prefix_x.shape, suffix_x.shape
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = indices
        ctx.save_for_backward(*indices)
        ctx.prefix_shape, ctx.suffix_shape = prefix_shape, suffix_shape
        # Concat the prefix and suffix inputs into a single grouped input tensor
        x = torch.zeros(
            *x_shape[:2],
            *prefix_shape[2:],
            dtype=prefix_x.dtype,
            device=prefix_x.device,
        )
        x[grouped_prefix_indices[:, 0], grouped_prefix_indices[:, 1]] = prefix_x[
            ungrouped_prefix_indices[:, 0], ungrouped_prefix_indices[:, 1]
        ]
        x[grouped_suffix_indices[:, 0], grouped_suffix_indices[:, 1]] = suffix_x[
            ungrouped_suffix_indices[:, 0], ungrouped_suffix_indices[:, 1]
        ]
        return x

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):
        (
            ungrouped_prefix_indices,
            ungrouped_suffix_indices,
            grouped_prefix_indices,
            grouped_suffix_indices,
        ) = ctx.saved_tensors
        prefix_shape, suffix_shape = ctx.prefix_shape, ctx.suffix_shape
        # Split the grad into prefix grad and suffix grad
        grad_prefix_x = torch.zeros(
            prefix_shape, dtype=grad_x.dtype, device=grad_x.device
        )
        grad_prefix_x[
            ungrouped_prefix_indices[:, 0], ungrouped_prefix_indices[:, 1]
        ] = grad_x[grouped_prefix_indices[:, 0], grouped_prefix_indices[:, 1]]
        grad_suffix_x = torch.zeros(
            suffix_shape, dtype=grad_x.dtype, device=grad_x.device
        )
        grad_suffix_x[
            ungrouped_suffix_indices[:, 0], ungrouped_suffix_indices[:, 1]
        ] = grad_x[grouped_suffix_indices[:, 0], grouped_suffix_indices[:, 1]]
        return grad_prefix_x, grad_suffix_x, None, None


class ConvertPaddingFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,  # NOTE: Shape: [b, seq, ...]
        indices: Tuple[torch.Tensor, torch.Tensor],  # 2 non-zero mask index tensors
        o_shape: torch.Size,  # shape of converted output tensor
    ):
        input_shape = x.shape
        ctx.input_shape = input_shape
        x_indices, o_indices = indices
        ctx.save_for_backward(*indices)
        o = torch.zeros(
            *o_shape[:2],
            *input_shape[2:],
            dtype=x.dtype,
            device=x.device,
        )
        o[o_indices[:, 0], o_indices[:, 1]] = x[x_indices[:, 0], x_indices[:, 1]]
        return o

    @staticmethod
    def backward(ctx, grad_o: torch.Tensor):
        x_indices, o_indices = ctx.saved_tensors
        grad_x = torch.zeros(
            ctx.input_shape,
            dtype=grad_o.dtype,
            device=grad_o.device,
        )
        grad_x[x_indices[:, 0], x_indices[:, 1]] = grad_o[o_indices[:, 0], o_indices[:, 1]]
        return grad_x, None, None
