import torch
from .utils import batch_repeat_cat
from .utils.typing import List, Union, Sequence, SupportsIndex
from .utils.mask import create_mask, create_submask, create_padding_mask


class Info(Sequence[int]):
    def __init__(self, prefix_len: int, suffix_lens: List[int]):
        assert len(suffix_lens) > 0, "Size of ``suffix_lens`` should be greater than 0"
        self.prefix_len = prefix_len
        self.suffix_lens = suffix_lens

    @property
    def num_samples(self) -> int:
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
        padding_mode: Union[str, torch.Tensor] = "right",
    ):
        # NOTE: The ``device`` is not assigned to ``self``, because the actual device may change
        # among different decoder layers
        assert len(info_list) > 0, "Size of ``info_list`` should be greater than 0"
        self.info_list = info_list
        self.precompute(device, padding_mode)

    @property
    def batch_size(self) -> int:
        return len(self.info_list)

    @classmethod
    def from_list(
        cls,
        group_info: List[List[int]],
        device=None,
        padding_mode: Union[str, torch.Tensor] = "right",
    ):
        return cls(
            [Info(prefix_len=g[0], suffix_lens=g[1:]) for g in group_info],
            device=device,
            padding_mode=padding_mode,
        )

    def precompute(
        self, device, padding_mode: Union[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Precompute intermediate cache variables.
        """
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
        self.num_samples = torch.tensor(
            [info.num_samples for info in self.info_list],
            dtype=torch.long,
            device=device,
        )
        self.total_lens = torch.tensor(
            [info.total_len for info in self.info_list], device=device
        )
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
        # Ungrouped Suffix Mask [num_samples, max_suffix_len]
        self.ungrouped_suffix_mask = create_mask(
            self.ungrouped_suffix_lens,
            max_len=int(self.ungrouped_suffix_lens.max().item()),
            padding_mode="right",
            device=device,
        )
        # Attention Mask
        self.prefix_attn_mask = self.ungrouped_prefix_mask
        self.suffix_attn_mask = batch_repeat_cat(
            self.ungrouped_prefix_mask,
            self.ungrouped_suffix_mask,
            cat_dim=1,
            num_samples=self.num_samples,
        )
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
