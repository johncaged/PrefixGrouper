from abc import ABC, abstractmethod
import torch
from .utils.typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from . import PrefixGrouper


class AttnFuncType(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        pass


class AttentionForwardABC(ABC):
    @abstractmethod
    def __call__(
        self,
        prefix_grouper: "PrefixGrouper",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Call the original attention function.
        """
        pass


class AttentionForward(AttentionForwardABC):
    def __init__(self, attn_func: AttnFuncType):
        """
        Apply attention forward using ``attn_func``.

        NOTE: the ``attn_func`` should accept q, k, v and attn_mask as the first 4 positional arguments.
        """
        super().__init__()
        self.attn_func = attn_func

    def __call__(
        self,
        prefix_grouper: "PrefixGrouper",
        # NOTE: the following are the original params needed in ``attn_func``
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        q_prefix, k_prefix, v_prefix, q_suffix, k_suffix, v_suffix = (
            prefix_grouper.ungroup(q, k, v)
        )

        prefix_attn_output = self.attn_func(
            q_prefix,
            k_prefix,
            v_prefix,
            prefix_grouper.group_info.prefix_attn_mask.to(q_prefix.device),
            *args,
            **kwargs,
        )
        suffix_attn_output = self.attn_func(
            q_suffix,
            prefix_grouper.batch_repeat_cat(k_prefix, k_suffix, cat_dim=2),
            prefix_grouper.batch_repeat_cat(v_prefix, v_suffix, cat_dim=2),
            prefix_grouper.group_info.suffix_attn_mask.to(q_suffix.device),
            *args,
            **kwargs,
        )
        attn_output = prefix_grouper.group(prefix_attn_output, suffix_attn_output)
        return attn_output
