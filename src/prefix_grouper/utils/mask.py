import torch
from .typing import Optional, overload, Union

SUPPORTED_PADDING_MODES = ["left", "right"]


def _resolve_start_end(
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
    end_indices: torch.Tensor,
    *,
    max_len: int,
    seq_len: Optional[torch.Tensor] = None,
    padding_mode: str = "right",
    device: torch.device = None,
) -> torch.Tensor: ...
@overload
def create_mask(
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
    *,
    max_len: int,
    seq_len: Optional[torch.Tensor] = None,
    padding_mode: str = "right",
    device: torch.device = None,
) -> torch.Tensor: ...
def create_mask(
    indices1: torch.Tensor,
    indices2: Optional[torch.Tensor] = None,
    *,
    max_len: int,
    seq_len: Optional[torch.Tensor] = None,
    padding_mode: str = "right",
    device: torch.device = None,
) -> torch.Tensor:
    """
    create mask based on padding mode
    """
    assert (
        padding_mode in SUPPORTED_PADDING_MODES
    ), f"Supported padding modes: {SUPPORTED_PADDING_MODES}, but got {padding_mode}"
    start_indices, end_indices = _resolve_start_end(
        indices1=indices1, indices2=indices2
    )
    if padding_mode == "left":
        if seq_len is None:
            raise ValueError(
                "``seq_len`` cannot be ``None`` when ``padding_mode`` is left"
            )
        padding_delta = max_len - seq_len
        start_indices = start_indices + padding_delta
        end_indices = end_indices + padding_delta
    positions = torch.arange(max_len, device=device)
    mask = (positions < end_indices.unsqueeze(-1)) & (
        positions >= start_indices.unsqueeze(-1)
    )
    return mask.bool()


@overload
def create_submask(
    mask: torch.Tensor,
    end_indices: torch.Tensor,
) -> torch.Tensor: ...
@overload
def create_submask(
    mask: torch.Tensor,
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
) -> torch.Tensor: ...
def create_submask(
    mask: torch.Tensor,
    indices1: torch.Tensor,
    indices2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Create submask based on the given mask and indices.
    """
    start_indices, end_indices = _resolve_start_end(
        indices1=indices1, indices2=indices2
    )
    counts = mask.long().cumsum(dim=1)
    index_tensor = torch.where(mask, counts - 1, torch.full_like(counts, -1))
    start_expanded = start_indices.unsqueeze(-1)
    end_expanded = end_indices.unsqueeze(-1)
    new_mask = (index_tensor >= start_expanded) & (index_tensor < end_expanded)
    return new_mask.bool()


def create_padding_mask(
    padding_mode: Union[str, torch.Tensor],
    total_lens: torch.Tensor,
    batch_size: int,
    device,
) -> torch.Tensor:
    """
    Create and verify padding masks. ``True`` represents non-padding tokens, while ``False``
    represents padding tokens.
    """
    assert (
        isinstance(padding_mode, str) and padding_mode in SUPPORTED_PADDING_MODES
    ) or isinstance(
        padding_mode, torch.Tensor
    ), f"``padding_mode`` should either be a ``str`` (supported values: {SUPPORTED_PADDING_MODES}) or a ``torch.Tensor`` mask."
    if isinstance(padding_mode, str):
        padding_mask = create_mask(
            total_lens,
            max_len=int(total_lens.max().item()),
            seq_len=total_lens,
            padding_mode=padding_mode,
            device=device,
        )
    else:
        padding_mask = padding_mode.to(device)
    # Verify padding mask
    assert (
        padding_mask.ndim == 2
    ), f"Padding mask should be a Tensor of shape [b, seq_len] (ndim == 2), got {padding_mask.shape}"
    assert padding_mask.shape[0] == (
        batch_size
    ), f"Size of padding mask at dim 0 should be equal to ``batch_size``, got {padding_mask.shape[0]} and {batch_size}"
    token_cnt = padding_mask.sum(dim=-1)
    assert torch.all(
        token_cnt == total_lens
    ), f"Number of True values in padding mask does not match ``total_lens``, got {token_cnt} and {total_lens}"
    return padding_mask.bool()
