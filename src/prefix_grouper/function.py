"""
PyTorch autograd functions.
"""

import torch
from torch.autograd import Function


class UngroupFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,  # NOTE: Shape: [b, num_heads, seq, head_dim]
        indices: tuple,  # 4 non-zero mask indices
        shapes: tuple,  # shapes of ungrouped prefix and ungrouped suffix
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
        ctx.save_for_backward(*indices)
        ctx.input_shape = input_shape
        # Split the grouped inputs into prefix and suffix tensors.
        prefix_x = torch.zeros(
            prefix_x_shape[0],
            input_shape[1],
            prefix_x_shape[1],
            input_shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        suffix_x = torch.zeros(
            suffix_x_shape[0],
            input_shape[1],
            suffix_x_shape[1],
            input_shape[-1],
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
        prefix_x: torch.Tensor,  # NOTE: Shape [b, seq, num_heads, head_dim]
        suffix_x: torch.Tensor,  # NOTE: Shape [b, seq, num_heads, head_dim]
        indices: tuple,  # 4 non-zero mask indices
        x_shape: torch.Tensor,  # shape of grouped input x
    ):
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
            x_shape[0],
            x_shape[1],
            prefix_shape[-2],
            prefix_shape[-1],
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
