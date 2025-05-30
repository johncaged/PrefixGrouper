import torch
from .. import PrefixGrouper
from ..forward import AttentionForward
from .typing import Optional


def _flash_attention_2_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    # The following are custom args
    module: torch.nn.Module,
    *args,
    **kwargs,
):
    # NOTE: we got q, k, v, attn_mask as the first 4 parameters, while ``flash_attn``
    # requires ``module`` as the first parameter, so we should write an adapter here.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    flash_attn_forward = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    return flash_attn_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        *args,
        **kwargs,
    )


def _grouped_flash_attention_2(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *args,
    prefix_grouper: PrefixGrouper,
    **kwargs,
):
    """
    Convert the attention call to ``AttentionForward``
    """
    # NOTE: ``attention_mask`` param is ignored.
    return AttentionForward(_flash_attention_2_forward)(
        prefix_grouper,
        query,
        key,
        value,
        # The following are custom parameters
        module,
        *args,
        **kwargs,
    )


def register_attention():
    """
    Register attention interface in ``transformers``
    """
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register("grouped_flash_attention_2", _grouped_flash_attention_2)
