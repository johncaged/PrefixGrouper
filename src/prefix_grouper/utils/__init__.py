import torch


def batch_repeat_cat(
    prefix: torch.Tensor, suffix: torch.Tensor, cat_dim: int, num_samples: torch.Tensor
) -> torch.Tensor:
    """
    Repeat the prefix tensor according to ``num_samples``, and cat it to the
    suffix tensor. NOTE: The tensor should be batch-first.
    """
    return torch.cat(
        [
            prefix.repeat_interleave(
                num_samples.to(prefix.device), dim=0
            ),  # batch repeat
            suffix,
        ],
        dim=cat_dim,
    )
