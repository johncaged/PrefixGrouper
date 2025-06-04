import torch
from .. import PrefixGrouper
from ..forward import AttentionForward
from .typing import Optional


def _attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    # The following are custom args
    module: torch.nn.Module,
    *args,
    prefix_grouper_attn_func: str = "flash_attention_2",
    **kwargs,
):
    # NOTE: we got q, k, v, attn_mask as the first 4 parameters, while ``flash_attn``
    # requires ``module`` as the first parameter, so we should write an adapter here.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    attn_forward = ALL_ATTENTION_FUNCTIONS[prefix_grouper_attn_func]
    # NOTE: We do not support returning attention weights for now
    return attn_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        *args,
        **kwargs,
    )[0]


def _prefix_grouper_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *args,
    prefix_grouper: PrefixGrouper,
    prefix_grouper_attn_func: str = "flash_attention_2",
    **kwargs,
):
    """
    Convert the attention call to ``AttentionForward``
    """
    # NOTE: ``attention_mask`` param is ignored.
    return (
        AttentionForward(_attention_forward)(
            prefix_grouper,
            query,
            key,
            value,
            # The following are custom parameters
            module,
            *args,
            prefix_grouper_attn_func=prefix_grouper_attn_func,
            **kwargs,
        ),
        None,
    )  # NOTE: We do not support returning attention weights for now.


def register_attention():
    """
    Register attention interface in ``transformers``
    """
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(
        "prefix_grouper_attention", _prefix_grouper_attention_forward
    )
