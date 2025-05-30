from abc import ABC, abstractmethod
from typing import List, Optional, overload, Union, Tuple
import torch
from .function import GroupFunction, UngroupFunction
from .forward import AttentionForward, AttnFuncType

SUPPORTED_PADDING_MODES = ["left", "right"]
UngroupedTuple = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class PrefixGrouperABC(ABC):
    @abstractmethod
    def precompute(self, *args, **kwargs):
        """
        Precompute any needed intermediate variables (e.g., attention mask, indices, etc.)
        """
        pass


class PrefixGrouper(PrefixGrouperABC):
    def __init__(
        self,
        group_info: List[List[int]],
        device=None,
        padding_mode: Union[str, torch.Tensor] = "right",
    ) -> None:
        # NOTE: The ``device`` is not assigned to ``self``, because the actual device may change
        # among different decoder layers
        for g in group_info:
            assert (
                len(g) >= 2
            ), f"Size of each element in ``group_info`` should be greater than 2"
        assert (
            isinstance(padding_mode, str) and padding_mode in SUPPORTED_PADDING_MODES
        ) or isinstance(
            padding_mode, torch.Tensor
        ), f"``padding_mode`` should either be a ``str`` (supported values: {SUPPORTED_PADDING_MODES}) or a ``torch.Tensor`` mask."
        self.group_info = group_info
        self.precompute(device, padding_mode)

    def ungroup(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> UngroupedTuple:
        """
        Ungroup the input tensors according to the ``group_info``.

        Input: q, k, v tensors in the shape of [b, num_heads, seq, head_dim]

        Output: q_prefix, k_prefix, v_prefix, q_suffix, k_suffix, v_suffix

        NOTE: You should carefully check the input and output shapes.
        """
        prefix_x_shape = self.prefix_x_shape
        suffix_x_shape = self.suffix_x_shape
        indices = (
            self.ungrouped_prefix_indices.to(q.device),
            self.ungrouped_suffix_indices.to(q.device),
            self.grouped_prefix_indices.to(q.device),
            self.grouped_suffix_indices.to(q.device),
        )
        shapes = (prefix_x_shape, suffix_x_shape)
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
        return GroupFunction.apply(
            o_prefix,
            o_suffix,
            (
                self.ungrouped_prefix_indices.to(o_prefix.device),
                self.ungrouped_suffix_indices.to(o_prefix.device),
                self.grouped_prefix_indices.to(o_prefix.device),
                self.grouped_suffix_indices.to(o_prefix.device),
            ),
            self.x_shape,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_func: AttnFuncType,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return AttentionForward(attn_func)(
            q,
            k,
            v,
            *args,
            prefix_grouper=self,
            **kwargs,
        )

    def precompute(
        self, device, padding_mode: Union[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Create 4 mask matrices, 4 indice matrices based on group_info.
        """
        prefix_elements = [g[0] for g in self.group_info]
        suffix_elements = []
        num_samples = []  # Save num samples in each group (for repeat interleave)
        for g in self.group_info:
            suffix_elements.extend(g[1:])
            num_samples.append(len(g) - 1)
        self.num_samples = torch.tensor(num_samples, dtype=torch.long, device=device)
        sums = torch.tensor([sum(g) for g in self.group_info], device=device)
        self.max_total_len: int = int(sums.max().item())

        # First, process grouped indices
        if isinstance(padding_mode, str):
            # Calculate ``padding_delta`` for left padding
            padding_delta = self.max_total_len - sums
            padding_mask = self.create_mask(
                sums,
                max_len=self.max_total_len,
                padding_mode=padding_mode,
                device=device,
                padding_delta=padding_delta,
            )
        else:
            padding_mask = padding_mode.to(device)
        # Verify padding mask
        assert (
            padding_mask.ndim == 2
        ), f"Padding mask should be a Tensor of shape [b, seq_len] (ndim == 2), got {padding_mask.shape}"
        assert padding_mask.shape[0] == len(
            self.group_info
        ), f"Padding mask should be the same size as ``group_info`` at dim 0, got {padding_mask.shape[0]} and {len(self.group_info)}"
        token_cnt = padding_mask.sum(dim=-1)
        assert torch.all(
            token_cnt == sums
        ), f"Number of True values in padding mask does not match ``group_info``, got {token_cnt} and {sums}"
        # NOTE: The ``padding_mask`` can be used as ``attention_mask`` in the model forward process.
        self.padding_mask = padding_mask
        # Grouped Prefix Mask [num_groups, max_total_len]
        grouped_prefix_mask = self.create_submask(
            padding_mask,
            torch.tensor(prefix_elements, device=device),
        )
        # Grouped Suffix Mask [num_groups, max_total_len]
        starts = torch.tensor(prefix_elements, device=device)
        lengths = torch.tensor([sum(g[1:]) for g in self.group_info], device=device)
        ends = starts + lengths
        grouped_suffix_mask = self.create_submask(
            padding_mask,
            starts,
            ends,
        )
        self.grouped_prefix_indices = grouped_prefix_mask.nonzero(as_tuple=False).to(
            device
        )
        self.grouped_suffix_indices = grouped_suffix_mask.nonzero(as_tuple=False).to(
            device
        )

        # Second, process ungrouped indices
        # NOTE: Ungrouped prefix and suffix are always right-padding, because it doesn't
        # matter whether it's left-padding or right-padding in the attention operations,
        # so we keep it right-padding for consistency and avoiding potential bugs.
        # Ungrouped Prefix Mask [num_groups, max_prefix_len]
        self.max_prefix_len: int = max(prefix_elements) if prefix_elements else 0
        ungrouped_prefix_mask: torch.Tensor = self.create_mask(
            torch.tensor(prefix_elements, device=device),
            max_len=self.max_prefix_len,
            padding_mode="right",
            device=device,
        )
        # Ungrouped Suffix Mask [num_samples, max_suffix_len]
        self.max_suffix_len: int = max(suffix_elements) if suffix_elements else 0
        ungrouped_suffix_mask: torch.Tensor = self.create_mask(
            torch.tensor(suffix_elements, device=device),
            max_len=self.max_suffix_len,
            padding_mode="right",
            device=device,
        )
        # Cache indices
        # Tuple[batch_dim, seq_dim]
        self.ungrouped_prefix_indices = ungrouped_prefix_mask.nonzero(
            as_tuple=False
        ).to(device)
        self.ungrouped_suffix_indices = ungrouped_suffix_mask.nonzero(
            as_tuple=False
        ).to(device)

        # x shape
        self.x_shape = grouped_prefix_mask.shape
        self.prefix_x_shape = ungrouped_prefix_mask.shape
        self.suffix_x_shape = ungrouped_suffix_mask.shape
        # Attention Mask
        self.prefix_attn_mask = ungrouped_prefix_mask.bool()
        self.suffix_attn_mask = self.batch_repeat_cat(
            ungrouped_prefix_mask, ungrouped_suffix_mask, cat_dim=1
        ).bool()

    def batch_repeat_cat(
        self, prefix: torch.Tensor, suffix: torch.Tensor, cat_dim: int
    ) -> torch.Tensor:
        """
        Repeat the prefix tensor according to ``num_samples``, and cat it to the
        suffix tensor. NOTE: The tensor should be batch-first.
        """
        return torch.cat(
            [
                prefix.repeat_interleave(
                    self.num_samples.to(prefix.device), dim=0
                ),  # batch repeat
                suffix,
            ],
            dim=cat_dim,
        )

    def _resolve_start_end(
        self,
        indices1: torch.Tensor,
        indices2: Optional[torch.Tensor] = None,
    ):
        """
        If ``indices2`` is ``None``, then set ``start_indices`` to 0, and
        set ``end_indices`` to ``indices1``.
        """
        if indices2 is not None:
            start_indices = indices1
            end_indices = indices2
        else:
            start_indices = torch.zeros_like(indices1)
            end_indices = indices1
        return start_indices, end_indices

    @overload
    def create_mask(
        self,
        end_indices: torch.Tensor,
        *,
        max_len: int,
        padding_mode: str = "right",
        device: torch.device = None,
        padding_delta: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor: ...
    @overload
    def create_mask(
        self,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
        *,
        max_len: int,
        padding_mode: str = "right",
        device: torch.device = None,
        padding_delta: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor: ...
    def create_mask(
        self,
        indices1: torch.Tensor,
        indices2: Optional[torch.Tensor] = None,
        *,
        max_len: int,
        padding_mode: str = "right",
        device: torch.device = None,
        padding_delta: Union[
            int, torch.Tensor
        ] = 0,  # NOTE: Used when ``padding_mode`` is "left"
    ) -> torch.Tensor:
        """
        create mask based on padding mode
        """
        start_indices, end_indices = self._resolve_start_end(
            indices1=indices1, indices2=indices2
        )
        if padding_mode == "left":
            start_indices = start_indices + padding_delta
            end_indices = end_indices + padding_delta
        positions = torch.arange(max_len, device=device)
        mask = (positions < end_indices.unsqueeze(-1)) & (
            positions >= start_indices.unsqueeze(-1)
        )
        return mask.int()

    @overload
    def create_submask(
        self,
        mask: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> torch.Tensor: ...
    @overload
    def create_submask(
        self,
        mask: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> torch.Tensor: ...
    def create_submask(
        self,
        mask: torch.Tensor,
        indices1: torch.Tensor,
        indices2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create submask based on the given mask and indices.
        """
        start_indices, end_indices = self._resolve_start_end(
            indices1=indices1, indices2=indices2
        )
        counts = mask.long().cumsum(dim=1)
        index_tensor = torch.where(mask, counts - 1, torch.full_like(counts, -1))
        start_expanded = start_indices.unsqueeze(-1)
        end_expanded = end_indices.unsqueeze(-1)
        new_mask = (index_tensor >= start_expanded) & (index_tensor < end_expanded)
        return new_mask
