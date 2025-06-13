import torch
from .utils import batch_repeat_cat
from .utils.typing import List, Union, Tuple, Sequence, SupportsIndex
from .utils.mask import create_mask, create_submask, create_padding_mask


class Info(Sequence[int]):
    def __init__(self, prefix_len: int, suffix_lens: List[int]):
        assert len(suffix_lens) > 0, "Size of ``suffix_lens`` should be greater than 0"
        self.prefix_len = prefix_len
        self.suffix_lens = suffix_lens

    @property
    def num_suffixes(self) -> int:
        return len(self.suffix_lens)

    @property
    def total_len(self) -> int:
        return self.prefix_len + sum(self.suffix_lens)

    def __getitem__(self, __index: Union[SupportsIndex, slice]):
        # NOTE: This is for backward compatibility, and is a low-efficiency implementation
        return [self.prefix_len, *self.suffix_lens][__index]

    def __len__(self) -> int:
        # NOTE: This is for backward compatibility
        return 1 + len(self.suffix_lens)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(prefix_len={self.prefix_len}, suffix_lens={self.suffix_lens})"

    def __repr__(self) -> str:
        return self.__str__()


class GroupInfo(Sequence[Info]):
    def __init__(
        self,
        info_list: List[Info],
        device=None,
        padding_mode: Union[str, torch.Tensor, None] = "right",
    ):
        assert len(info_list) > 0, "Size of ``info_list`` should be greater than 0"
        self.info_list = info_list
        self._init_device = device
        self._padding_mode = padding_mode
        self.precompute()

    @property
    def batch_size(self) -> int:
        return len(self.info_list)

    @classmethod
    def from_list(
        cls,
        group_info: List[List[int]],
        device=None,
        padding_mode: Union[str, torch.Tensor, None] = "right",
    ):
        return cls(
            [Info(prefix_len=g[0], suffix_lens=g[1:]) for g in group_info],
            device=device,
            padding_mode=padding_mode,
        )

    def precompute(self) -> List[torch.Tensor]:
        """
        Precompute intermediate cache variables.
        """
        self.precompute_sizes()
        self.precompute_tensor_masks()
        self.precompute_attn_masks()
        self.precompute_indices()
        self.precompute_shapes()

    def precompute_sizes(self):
        device = self._init_device
        self.prefix_lens = torch.tensor(
            [info.prefix_len for info in self.info_list],
            dtype=torch.long,
            device=device,
        )
        self.grouped_suffix_lens = torch.tensor(
            [sum(info.suffix_lens) for info in self.info_list],
            dtype=torch.long,
            device=device,
        )
        self.ungrouped_suffix_lens = torch.tensor(
            [suffix_len for info in self.info_list for suffix_len in info.suffix_lens],
            dtype=torch.long,
            device=device,
        )
        self.num_suffixes = torch.tensor(
            [info.num_suffixes for info in self.info_list],
            dtype=torch.long,
            device=device,
        )
        self.total_lens = torch.tensor(
            [info.total_len for info in self.info_list], device=device
        )

    def precompute_tensor_masks(self):
        device = self._init_device
        padding_mode = self._padding_mode
        self.padding_mask = create_padding_mask(
            padding_mode=padding_mode,
            total_lens=self.total_lens,
            batch_size=self.batch_size,
            device=device,
        )
        # Grouped Prefix Mask [num_groups, max_total_len]
        self.grouped_prefix_mask = create_submask(self.padding_mask, self.prefix_lens)
        # Grouped Suffix Mask [num_groups, max_total_len]
        self.grouped_suffix_mask = create_submask(
            self.padding_mask,
            self.prefix_lens,
            self.prefix_lens + self.grouped_suffix_lens,
        )
        # NOTE: Ungrouped prefix is always left-padding and suffix is always right-padding,
        # because it doesn't matter whether it's left-padding or right-padding in the
        # attention operations, so we choose to have no padding between the prefix and suffix
        # for consistency and convenience.
        # Ungrouped Prefix Mask [num_groups, max_prefix_len]
        self.ungrouped_prefix_mask = create_mask(
            self.prefix_lens,
            max_len=int(self.prefix_lens.max().item()),
            seq_len=self.prefix_lens,
            padding_mode="left",
            device=device,
        )
        # Ungrouped Suffix Mask [num_suffixes, max_suffix_len]
        self.ungrouped_suffix_mask = create_mask(
            self.ungrouped_suffix_lens,
            max_len=int(self.ungrouped_suffix_lens.max().item()),
            padding_mode="right",
            device=device,
        )

    def precompute_attn_masks(self):
        # Attention Mask
        self.prefix_attn_mask = self.ungrouped_prefix_mask
        self.suffix_attn_mask = batch_repeat_cat(
            self.ungrouped_prefix_mask,
            self.ungrouped_suffix_mask,
            cat_dim=1,
            num_suffixes=self.num_suffixes,
        )

    def precompute_indices(self):
        device = self._init_device
        # Cache indices
        # Tuple[batch_dim, seq_dim]
        self.grouped_prefix_indices = self.grouped_prefix_mask.nonzero(
            as_tuple=False
        ).to(device)
        self.grouped_suffix_indices = self.grouped_suffix_mask.nonzero(
            as_tuple=False
        ).to(device)
        self.ungrouped_prefix_indices = self.ungrouped_prefix_mask.nonzero(
            as_tuple=False
        ).to(device)
        self.ungrouped_suffix_indices = self.ungrouped_suffix_mask.nonzero(
            as_tuple=False
        ).to(device)

    def precompute_shapes(self):
        # Cache input shapes
        self.x_shape = self.padding_mask.shape
        self.prefix_x_shape = self.ungrouped_prefix_mask.shape
        self.suffix_x_shape = self.ungrouped_suffix_mask.shape

    def __getitem__(self, __index: Union[SupportsIndex, slice]):
        # NOTE: For backward compatibility
        return self.info_list[__index]

    def __len__(self) -> int:
        # NOTE: This is for backward compatibility
        return len(self.info_list)

    def __str__(self) -> str:
        return str(self.info_list)

    def __repr__(self) -> str:
        return self.__str__()


class GroupInfoForPackedSequence(GroupInfo):
    def __init__(
        self,
        info_list: List[Info],
        device=None,
        padding_mode: Union[torch.Tensor, None] = None,
    ):
        assert (padding_mode is None) or (
            isinstance(padding_mode, torch.Tensor) and padding_mode.ndim == 1
        ), f"``padding_mode`` can only be ``None`` or a 1d ``torch.Tensor`` in ``GroupInfoForPackedSequence``"
        super().__init__(info_list, device=device, padding_mode=padding_mode)

    @classmethod
    def from_list(
        cls,
        group_info: List[List[int]],
        device=None,
        padding_mode: Union[torch.Tensor, None] = None,
    ):
        return super().from_list(group_info, device=device, padding_mode=padding_mode)

    def precompute_tensor_masks(self):
        device = self._init_device
        padding_mode: Union[torch.Tensor, None] = self._padding_mode
        self.padding_mask = (
            torch.ones(
                int(self.total_lens.sum().item()), dtype=torch.bool, device=device
            )
            if padding_mode is None
            else padding_mode
        )
        assert (
            self.padding_mask.sum() == self.total_lens.sum()
        ), f"Valid tokens computed by ``padding_mask`` is inconsistent with ``info_list``. Got {self.padding_mask.sum()} and {self.total_lens.sum()}"
        # Grouped Masks
        self.grouped_prefix_mask, self.grouped_suffix_mask = (
            self._precompute_ungrouped_masks()
        )
        # NOTE: Ungrouped prefix is always left-padding and suffix is always right-padding,
        # because it doesn't matter whether it's left-padding or right-padding in the
        # attention operations, so we choose to have no padding between the prefix and suffix
        # for consistency and convenience.
        # Ungrouped Prefix Mask [num_groups, max_prefix_len]
        self.ungrouped_prefix_mask = create_mask(
            self.prefix_lens,
            max_len=int(self.prefix_lens.max().item()),
            seq_len=self.prefix_lens,
            padding_mode="left",
            device=device,
        )
        # Ungrouped Suffix Mask [num_suffixes, max_suffix_len]
        self.ungrouped_suffix_mask = create_mask(
            self.ungrouped_suffix_lens,
            max_len=int(self.ungrouped_suffix_lens.max().item()),
            padding_mode="right",
            device=device,
        )

    def precompute_indices(self):
        super().precompute_indices()
        # NOTE: Compute packed_q_indices, cu_seq_lens_q, packed_kv_indices, cu_seq_lens_kv
        device = self._init_device
        # q indices and cu_seq_lens
        self.packed_q_indices = self.padding_mask.nonzero(as_tuple=False)
        self.cu_seq_lens_q = torch.tensor(
            [0, *(i for info in self.info_list for i in info)],
            dtype=torch.long,
            device=device,
        ).cumsum(0)
        # k/v indices and cu_seq_lens
        self.cu_seq_lens_kv = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                *(
                    torch.tensor(
                        [
                            info.prefix_len,
                            *[(info.prefix_len + s_len) for s_len in info.suffix_lens],
                        ],
                        device=device,
                    )
                    for info in self.info_list
                ),
            ]
        ).cumsum(0)
        self.packed_kv_indices = self._precompute_packed_kv_indices()

    def _precompute_packed_kv_indices(self) -> torch.Tensor:
        all_valid_indices = self.padding_mask.nonzero(as_tuple=False)
        final_indices_list = []
        current_pos = 0
        for info in self.info_list:
            prefix_indices = all_valid_indices[
                current_pos : current_pos + info.prefix_len
            ]
            # Single Prefix k/v
            final_indices_list.append(prefix_indices)
            current_pos += info.prefix_len
            for suffix_len in info.suffix_lens:
                suffix_indices = all_valid_indices[
                    current_pos : current_pos + suffix_len
                ]
                # Concat Prefix + Suffix k/v
                combined_indices = torch.cat([prefix_indices, suffix_indices])
                final_indices_list.append(combined_indices)
                current_pos += suffix_len
        if current_pos != all_valid_indices.shape[0]:
            print("WARNING: inconsistency between ``padding_mask`` and ``info_list``")
        packed_kv_indices = torch.cat(final_indices_list)
        return packed_kv_indices

    def _precompute_ungrouped_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self._init_device
        all_valid_indices = torch.nonzero(self.padding_mask, as_tuple=False)[:, 0]
        seq_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                self.total_lens.cumsum(0)[:-1],
            ]
        )
        # Grouped Prefix Mask
        prefix_starts = seq_offsets
        prefix_ends = prefix_starts + self.prefix_lens
        grouped_prefix_mask = torch.zeros(
            self.padding_mask.shape[0], dtype=torch.bool, device=device
        )
        grouped_prefix_mask[
            torch.cat(
                [all_valid_indices[s:e] for s, e in zip(prefix_starts, prefix_ends)]
            )
        ] = True
        # Grouped Suffix Mask
        suffix_starts = prefix_ends
        suffix_ends = suffix_starts + self.grouped_suffix_lens
        grouped_suffix_mask = torch.zeros(
            self.padding_mask.shape[0], dtype=torch.bool, device=device
        )
        grouped_suffix_mask[
            torch.cat(
                [all_valid_indices[s:e] for s, e in zip(suffix_starts, suffix_ends)]
            )
        ] = True
        return grouped_prefix_mask, grouped_suffix_mask
