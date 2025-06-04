try:
    from .__version__ import __version__
except Exception:
    print("__version__ load failed.")

import torch
from .utils import batch_repeat_cat
from .utils.typing import List, Union, Tuple, Optional
from .function import GroupFunction, UngroupFunction
from .forward import AttentionForward, AttnFuncType
from .info import GroupInfo

UngroupedTuple = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
SplittedOutputTuple = Tuple[torch.Tensor, torch.Tensor]


class PrefixGrouper:
    def __init__(
        self,
        group_info: Optional[List[List[int]]] = None,
        device=None,
        padding_mode: Union[str, torch.Tensor] = "right",
    ) -> None:
        """
        NOTE: If ``group_info`` is None, then initialization is not performed, and you can
        call the ``init`` method later.
        """
        if group_info is not None:
            self.init(group_info, device, padding_mode)

    def init(
        self,
        group_info: List[List[int]],
        device=None,
        padding_mode: Union[str, torch.Tensor] = "right",
    ):
        if hasattr(self, "group_info"):
            print("WARNING: You are trying to re-init the ``group_info`` param.")
        self.group_info = GroupInfo.from_list(
            group_info=group_info, device=device, padding_mode=padding_mode
        )

    def get_ungroup_args(self, device=None):
        """
        Get ungroup indices and shapes.
        """
        prefix_x_shape = self.prefix_x_shape
        suffix_x_shape = self.suffix_x_shape
        indices = (
            self.ungrouped_prefix_indices.to(device),
            self.ungrouped_suffix_indices.to(device),
            self.grouped_prefix_indices.to(device),
            self.grouped_suffix_indices.to(device),
        )
        shapes = (prefix_x_shape, suffix_x_shape)
        return indices, shapes

    def ungroup(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> UngroupedTuple:
        """
        Ungroup the input tensors according to the ``group_info``.

        Input: q, k, v tensors in the shape of [b, num_heads, seq, head_dim]

        Output: q_prefix, k_prefix, v_prefix, q_suffix, k_suffix, v_suffix

        NOTE: You should carefully check the input and output shapes.
        """
        indices, shapes = self.get_ungroup_args(q.device)
        q_prefix, q_suffix = UngroupFunction.apply(q, indices, shapes)
        k_prefix, k_suffix = UngroupFunction.apply(k, indices, shapes)
        v_prefix, v_suffix = UngroupFunction.apply(v, indices, shapes)
        return q_prefix, k_prefix, v_prefix, q_suffix, k_suffix, v_suffix

    def group(self, o_prefix: torch.Tensor, o_suffix: torch.Tensor) -> torch.Tensor:
        """
        Pack the prefix and suffix attention outputs into a single tensor according to the
        ``group_info``.

        Input: o_prefix, o_suffix tensors in the shape of [b, seq, num_heads, head_dim]

        Output: a single attention output tensor in the shape of [b, seq, num_heads, head_dim]

        NOTE: You should carefully check the input and output shapes.
        """
        device = o_prefix.device
        return GroupFunction.apply(
            o_prefix,
            o_suffix,
            (
                self.ungrouped_prefix_indices.to(device),
                self.ungrouped_suffix_indices.to(device),
                self.grouped_prefix_indices.to(device),
                self.grouped_suffix_indices.to(device),
            ),
            self.x_shape,
        )

    def concat_input(
        self,
        prefix: torch.Tensor,
        prefix_mask: torch.Tensor,
        suffix: torch.Tensor,
        suffix_mask: torch.Tensor,
    ):
        """
        Concatenate the prefix and suffix inputs into grouped inputs based on the given masks
        and ``group_info``.

        Input: prefix, suffix tensors in the shape of [b, seq, ...]

        Output: input tensor in the shape of [b, seq, ...]
        """
        assert prefix.ndim == suffix.ndim
        assert (
            prefix.ndim == 2 or prefix.ndim == 3
        ), f"Can only accept input shape of [b, seq] or [b, seq, dim], got {prefix.shape}"

        device = prefix.device
        input_: torch.Tensor = GroupFunction.apply(
            # NOTE: unsqueeze to 4d to fit the group function
            prefix.reshape(*prefix.shape, *((1,) * (4 - prefix.ndim))),
            suffix.reshape(*suffix.shape, *((1,) * (4 - suffix.ndim))),
            (
                prefix_mask.nonzero(as_tuple=False).to(device),
                suffix_mask.nonzero(as_tuple=False).to(device),
                self.grouped_prefix_indices.to(device),
                self.grouped_suffix_indices.to(device),
            ),
            self.x_shape,
        )
        return input_.reshape(*(input_.shape[: prefix.ndim]))

    def split_output(
        self,
        output: torch.Tensor,
        include_prefix_last: int = 0,
    ) -> SplittedOutputTuple:
        """
        Split the output into prefix and suffix parts.

        Input: output tensors in the shape of [b, seq, dim]

        Output: prefix, suffix tensors in the shape of [b, seq, dim]
        """
        assert include_prefix_last >= 0
        assert (
            output.ndim == 3
        ), f"``output`` should be in the shape of [b, seq, dim], got {output.shape}"
        indices, shapes = self.get_ungroup_args(output.device)
        prefix_output, suffix_output = UngroupFunction.apply(
            output.unsqueeze(1),  # NOTE: unsqueeze to 4d to fit the ungroup function
            indices,
            shapes,
        )
        prefix_output, suffix_output = (
            prefix_output.squeeze(1),
            suffix_output.squeeze(1),
        )
        prefix_mask, suffix_mask = (
            self.ungrouped_prefix_mask,
            self.ungrouped_suffix_mask,
        )
        if include_prefix_last > 0:
            suffix_output = self.batch_repeat_cat(
                prefix_output[:, -include_prefix_last:], suffix_output, cat_dim=1
            )
            prefix_output = prefix_output[:, :-include_prefix_last]
            suffix_mask = self.batch_repeat_cat(
                prefix_mask[:, -include_prefix_last:], suffix_mask, cat_dim=1
            )
            prefix_mask = prefix_mask[:, :-include_prefix_last]
        return prefix_output, prefix_mask, suffix_output, suffix_mask

    def forward(
        self,
        __attn_func: AttnFuncType,
        # NOTE: the following are the original params needed in ``attn_func``
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return AttentionForward(__attn_func)(
            self,
            q,
            k,
            v,
            *args,
            **kwargs,
        )

    def batch_repeat_cat(
        self, prefix: torch.Tensor, suffix: torch.Tensor, cat_dim: int
    ) -> torch.Tensor:
        return batch_repeat_cat(
            prefix=prefix,
            suffix=suffix,
            cat_dim=cat_dim,
            num_samples=self.num_samples,
        )

    # NOTE: We manually set property here rather than using dynamic ``__getattribute__`` to enable type hint
    @property
    def prefix_lens(self):
        return self.group_info.prefix_lens

    @property
    def grouped_suffix_lens(self):
        return self.group_info.grouped_suffix_lens

    @property
    def ungrouped_suffix_lens(self):
        return self.group_info.ungrouped_suffix_lens

    @property
    def num_samples(self):
        return self.group_info.num_samples

    @property
    def total_lens(self):
        return self.group_info.total_lens

    @property
    def padding_mask(self):
        return self.group_info.padding_mask

    @property
    def grouped_prefix_mask(self):
        return self.group_info.grouped_prefix_mask

    @property
    def grouped_suffix_mask(self):
        return self.group_info.grouped_suffix_mask

    @property
    def ungrouped_prefix_mask(self):
        return self.group_info.ungrouped_prefix_mask

    @property
    def ungrouped_suffix_mask(self):
        return self.group_info.ungrouped_suffix_mask

    @property
    def prefix_attn_mask(self):
        return self.group_info.prefix_attn_mask

    @property
    def suffix_attn_mask(self):
        return self.group_info.suffix_attn_mask

    @property
    def grouped_prefix_indices(self):
        return self.group_info.grouped_prefix_indices

    @property
    def grouped_suffix_indices(self):
        return self.group_info.grouped_suffix_indices

    @property
    def ungrouped_prefix_indices(self):
        return self.group_info.ungrouped_prefix_indices

    @property
    def ungrouped_suffix_indices(self):
        return self.group_info.ungrouped_suffix_indices

    @property
    def x_shape(self):
        return self.group_info.x_shape

    @property
    def prefix_x_shape(self):
        return self.group_info.prefix_x_shape

    @property
    def suffix_x_shape(self):
        return self.group_info.suffix_x_shape
