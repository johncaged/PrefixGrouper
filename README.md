# PrefixGrouper

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/johncaged/PrefixGrouper/blob/main/i18n/README_zh-hans.md">简体中文</a>
    </p>
</h4>

``PrefixGrouper`` is a plug-and-play efficient GRPO training tool that requires minimal modifications to existing codebases to achieve reduced computation, lower VRAM consumption, and accelerated training. Additionally, this tool can be applied to other scenarios requiring shared-prefix training/inference beyond GRPO.

In current mainstream GRPO training pipelines, policy model training primarily involves copying prefixes (typically questions, multimodal inputs, etc.) `G` times. Consequently, when training data prefixes are sufficiently long (e.g., long-context reasoning, image/long-video inference), redundant computation during training becomes non-negligible, leading to increased VRAM usage, higher computation costs, and slower training speeds. To address this, we propose ``PrefixGrouper``, a plug-and-play GRPO training tool that enables efficient training through shared-prefix forward passes. Reduced VRAM consumption conversely allows more GPUs to support larger group sizes—critical for GRPO algorithms.

## News

- [2025/6/3] We release ``PrefixGrouper``. Tech report is coming, please stay tuned.

## Installation

```py
pip install prefix_grouper
```

## Quick Start

To make ``PrefixGrouper`` simpler and easier to use, we provide modification examples for some models.

- Model file modification examples can be found in ``examples``. For clarity, we wrap key modifications with "PrefixGrouper Start" and "PrefixGrouper End" comments.
- For examples simulating the full training workflow, see ``tests/equivalence``. We provide an almost complete flow for one training step.

If you happen to use one of these models, you can directly integrate the example code into your codebase. However, we recommend briefly reviewing the tutorial below to better understand the tool's workflow.

## Tutorial

Briefly, ``PrefixGrouper`` requires modifications in three areas: data input/output, attention mechanisms, and position encoding. Throughout this document, we refer to data corresponding to a query (prefix) as a **sample**, and each model-generated output based on the prefix as a **response**.

### Data Input/Output

To minimize redundant prefix forward passes and maximize parallel acceleration, ``PrefixGrouper`` first concatenates each sample in the batch with its corresponding responses (pseudocode example):

```py
# Assume we have prefix input_ids (padded torch.Tensor, shape [b, seq_len1]) and corresponding mask
prompt_ids = ...
prompt_mask = ...

# Assume model-generated responses are obtained (method irrelevant; output as str or input_ids)
# responses: List[List[str]] - outer list: batch_size (b), inner list: number of responses per sample.
responses: List[List[str]] = ...
# Flatten responses, tokenize, and pad into suffix_ids (shape [b * group_size, seq_len2])
suffix_ids = ...
suffix_mask = ...

# ====== Input processing complete ======

# Create PrefixGrouper instance
# group_info: List[List[int]] - outer list: sample count (b), inner list: [prefix_len, suffix1_len, suffix2_len,...]
group_info: List[List[int]] = ...
prefix_grouper = PrefixGrouper(
    group_info=group_info,
    padding_mode="right",
    device=device,
)
# Concatenate inputs into optimized input_ids (shape [b, seq_len])
input_ids = prefix_grouper.concat_input(prompt_ids, prompt_mask, suffix_ids, suffix_mask)
# Model forward with prefix_grouper
res = model(*args, **kwargs, prefix_grouper=prefix_grouper)
# ====== Forward pass complete ======

# Loss calculation and backward pass
# Split outputs (handles autoregressive boundary tokens)
prefix_output, prefix_mask, suffix_output, suffix_mask = (
    prefix_grouper.split_output(res.logits, include_prefix_last=1)
)
# ====== Output processing complete ======

# Standard GRPO loss calculation and backpropagation
suffix_output = suffix_output[:, :-1]
suffix_mask = suffix_mask[:, 1:]
loss = (suffix_output.gather(-1, suffix_ids.unsqueeze(-1)).squeeze(-1) - suffix_output.logsumexp(-1)).exp()
loss = loss * suffix_mask
loss = (loss.sum(-1) / suffix_mask.sum(-1)).mean()
(-loss).backward()
```

Key points for data processing: input concatenation, `group_info` statistics, and output splitting. Customize based on your project needs while maintaining interface consistency (see docs).

### Attention Mechanism

Minor model modifications suffice for attention adaptation. For transformers supporting ``AttentionInterface``, simple registration is possible (experimental). Below describes the generic approach:

```py
if prefix_grouper is None:
    # Original attention (baseline)
    attn_output = _flash_attention_forward(...)
else:
    # ===== PrefixGrouper Start =====
    def attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor, *args, **kwargs):
        # Adapter function for dimension/parameter alignment
        return _flash_attention_forward(...)
    
    attn_output = prefix_grouper.forward(...)
    # ====== PrefixGrouper End ======
```

Propagate `prefix_grouper` parameter through model forward passes.

### Position Encoding

Position IDs for concatenated responses should share the same starting ID pattern: ``[0, 1, ..., prefix_len, prefix_len+1, ..., suffix1_end, prefix_len+1, ..., suffix2_end, ...]``.

Position encoding is pre-adapted for models in the ``Quick Start`` section (see examples).

### Start Training

Complete GRPO training simulations are provided in ``tests`` for reference.

## Documentation

Core API documentation:

### PrefixGrouper

#### PrefixGrouper(group_info: List[List[int]], device=None, padding_mode: Union[str, torch.Tensor] = "right")

`group_info`: Outer list: sample count (b). Inner lists: [prefix_len, suffix1_len, suffix2_len,...].

`device`: Device for initializing PrefixGrouper (actual ops use input tensor's device).

`padding_mode`: `"left"`/`"right"` (dense padding) or `torch.Tensor` (custom padding mask, shape [b, seq_len]).

Usage examples:
- With `concat_input` (recommended):
```py
prefix_grouper = PrefixGrouper(group_info, padding_mode="right")
```
- Custom input handling:
```py
prefix_grouper = PrefixGrouper(group_info, padding_mode=custom_padding_mask)
```

#### PrefixGrouper.concat_input(self, prefix: torch.Tensor, prefix_mask: torch.Tensor, suffix: torch.Tensor, suffix_mask: torch.Tensor)

Concatenates `prefix` ([b, seq_len] or [b, seq_len, dim]) and `suffix` ([b*group_size, seq_len] or [b*group_size, seq_len, dim]) using `group_info`. Requires `prefix_mask`/`suffix_mask` (shape [b, seq_len]).

#### PrefixGrouper.forward(self, __attn_func: AttnFuncType, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs)

Performs attention using `__attn_func`. Function signature: `attn_func(q, k, v, attn_mask, *args, **kwargs)`. Input `q/k/v` shape: [b, num_heads, seq_len, head_dim]. Output shape: [b, seq_len, num_heads, head_dim]. **Do not manually pass attention masks.**

#### PrefixGrouper.split_output(self, output: torch.Tensor, include_prefix_last: int = 0)

`output`: Shape [b, seq_len, dim]
`include_prefix_last`: Controls prefix boundary handling (0: no conversion; 1: attach last prefix token to suffixes).

## Data Usage Statement

Test data in this project is strictly for **academic research purposes** with the following limitations:

1. **Commercial use is prohibited**  
2. **Data redistribution is prohibited**  
3. **De-anonymization attempts are prohibited**  

## Citation

[TODO]
