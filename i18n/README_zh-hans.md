<h3 align="center">
    <img src="https://raw.githubusercontent.com/johncaged/PrefixGrouper/main/assets/images/logo.png" width="352" style="max-width: 100%;">
</h3>

<h4 align="center">
    <p>
        <a href="https://github.com/johncaged/PrefixGrouper">English</a> |
        <b>简体中文</b>
    </p>
</h4>

<h3 align="center">
    <p>通过共享前缀前向传播实现高效GRPO训练</p>
</h3>

``PrefixGrouper`` 是一个即插即用的高效 GRPO 训练工具，只需在已有的训练代码库上进行少量的修改，即可实现计算量减少、显存占用降低、训练加速。此外，该工具还可以用到除 GRPO 之外的其他需要共享前缀训练 / 推理的场景。

目前主流的 GRPO 训练流程中，策略模型的训练主要是通过将前缀（通常是问题、多模态输入等）复制 ``G`` 次来实现的，那么显而易见的，当训练数据的前缀足够长时（如长上下文推理、图像或长视频推理等），训练过程中产生的冗余计算就变得不可忽视，从而导致显存占用增加、计算量增大、训练速度下降。对此，我们提出了 ``PrefixGrouper``，一个即插即用的 GRPO 训练工具，通过共享前缀的 forward 来实现高效训练。相对地，显存占用的降低反过来可以使得相同数量的显卡可以支持的最大 group_size 变大，这对于 GRPO 算法来说是至关重要的。

## 最新动态

**[2025/6/9]** 我们的技术报告已发布，<a href="https://arxiv.org/abs/2506.05433">点击此处查看</a>！

**[2025/6/7]** 我们更新 ``PrefixGrouper`` 版本到 ``0.0.1rc2``，封装性更好，代码改动更少，欢迎使用！

**[2025/6/3]** 我们正式发布 ``PrefixGrouper`` 工具。技术报告即将推出，敬请期待。

## 方法概览

``PrefixGrouper`` 的核心是其注意力操作的设计：

<h3 align="center">
    <img src="https://raw.githubusercontent.com/johncaged/PrefixGrouper/main/assets/images/method.jpg">
</h3>

通过将原始冗余的自注意力操作分解成前缀自注意力 + 后缀拼接注意力，``PrefixGrouper`` 可以实现高效 GRPO 训练，并且理论上兼容各种注意力实现（``EagerAttention``、``FlashAttention``、``SDPA`` 等）和各种硬件设备（GPU、NPU等）。

``PrefixGrouper`` 与 baseline FLOPs 和显存占用对比如下，展示了固定前缀长度下（4096、8192、16384）不同前后缀长度比例的结果（前缀长度 / 后缀长度）：

<h3 align="center">
    <img src="https://raw.githubusercontent.com/johncaged/PrefixGrouper/main/assets/images/flops.png" width="90%" style="max-width: 90%">
</h3>

<h3 align="center">
    <img src="https://raw.githubusercontent.com/johncaged/PrefixGrouper/main/assets/images/mem.png" width="90%" style="max-width: 90%">
</h3>

``PrefixGrouper`` 在长上下文场景下展现出明显优势，更说明其高效性。

## 安装

```bash
pip install prefix_grouper
```

## 快速上手

为了让 ``PrefixGrouper`` 更简单易用，我们提供了部分模型的修改示例。

- 模型文件修改示例请查看 ``examples``。为了更清晰，我们在关键修改部分采用 "PrefixGrouper Start" 和 "PrefixGrouper End" 注释包裹。
- 模拟整体训练流程的示例请查看 ``tests/equivalence``。我们提供一个训练 step 的几乎完整的流程。

如果你恰好需要使用示例中的模型进行训练，那么可以直接将示例中的代码引入到你的代码库。然而，我们还是建议你简单了解一下 ``PrefixGrouper`` 的使用教程，以更清楚地了解该工具的运行流程。

运行示例：

```bash
cd PrefixGrouper
python src/tests/equivalence/test_xxx.py --model_path /path/to/your/model
```

## 使用教程

简单来说，``PrefixGrouper`` 主要需要对 GRPO 训练流程进行三个方面的修改：数据输入输出、注意力机制和位置编码。在全文中，我们将一个问题 query（也就是前缀）所对应的数据称为一个 sample，而由模型根据前缀所采样生成的每一个输出，我们称为一个 response。

### 数据输入输出

为了降低前缀 forward 冗余、尽可能利用并行加速，``PrefixGrouper`` 首先将 batch 内的每一个 sample 与它所对应的多个 responses 进行拼接，得到 grouped 输入（以下进行伪代码示例）：

- 最佳实践（需要 ``0.0.1rc2`` 版本及以上）

```py
# 前缀：[b1, seq_len1]，b1 应为 sample 数量
prompt_ids = ...
# 前缀 mask：[b1, seq_len1]
prompt_mask = ...
# 后缀：[b2, seq_len2]，b2 应为所有 sample 的 responses 总数
completion_ids = ...
# 后缀 mask：[b2, seq_len2]
completion_mask = ...
# int 或 List[int]，int 代表每个 sample 都有相同数量的 responses，List[int] 则指定每个 sample 具有不同数量的 responses。
group_sizes = ...

# 初始化一个 PrefixGrouper 实例。
prefix_grouper = PrefixGrouper.from_ungrouped_masks(
    prefix_mask=prompt_mask,
    suffix_mask=completion_mask,
    group_sizes=group_sizes,
    padding_mode="right",
    device=device,
)
# 这里我们利用 PrefixGrouper 将分散的输入拼接成最终的 input_ids，其 shape 为 [b1, seq_len]。
# NOTE: 还可以输入特征，即 prompt_embeds ([b1, seq_len1, dim])，suffix_embeds ([b2, seq_len2, dim])
input_ids = prefix_grouper.concat_input(prompt_ids, prompt_mask, completion_ids, completion_mask)

# 这里进行模型 forward，只需要多加一个参数即可
res = model(*args, **kwargs, prefix_grouper=prefix_grouper)
# ====== 至此，forward 流程完毕 ======
# ``include_prefix_last`` 参数说明：注意到，response 的第一个 token 输出，实际上是由 prefix 最后一个 token 的输入产生的，因此 prefix 的最后一个 token 的输出需要计算 loss，那么 ``split_output`` 传入 ``include_prefix_last=1`` 参数，意味着 ``PrefixGrouper`` 会将 prefix 的最后一个 token 通过 repeat 的方式 concat 到 suffix 的最开头，得到的 mask 同样是进行过相同的处理的。
prefix_output, prefix_mask, suffix_output, suffix_mask = (
    prefix_grouper.split_output(res.logits, include_prefix_last=1)
)
# 这里必须将 completion_ids 转换为 right padding，以与 suffix_output 的位置对齐
completion_ids = prefix_grouper.convert_padding(completion_ids, completion_mask, padding_mode="right")
# ====== 至此，输入输出的流程全部完成 ======

# 获取了正常输出之后，接下来就可以按照 GRPO 来计算 loss、反向传播了，与标准 GRPO 流程完全一致。
# NOTE: 这里有所省略，比如 advantage、kl loss、importance sampling 等，请依照实际需要编写自己的 GRPO loss。
suffix_output = suffix_output[:, :-1]
suffix_mask = suffix_mask[:, 1:]
# NOTE: 由于 suffix_output 使用了 ``include_prefix_last=1``，因此 ``completion_ids`` 实际上比 ``suffix_output`` 短 1 个 token
# 因此它就不需要使用 [:, 1:] 了，因为第一个 token 开始就是有效 target。
loss = (suffix_output.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1) - suffix_output.logsumexp(-1)).exp()
loss = loss * suffix_mask
loss = (loss.sum(-1) / suffix_mask.sum(-1)).mean()
(-loss).backward()
```

- 旧版本示例：请查看 ``tests/test_equivalence``。

总之，数据输入输出处理的关键点是输入数据的拼接、group_info 的统计和输出的拆分，你完全可以根据实际项目的需要和模型的设计来自定义你的处理，只需要最终接口输入的参数一致即可（接口详见使用文档）。

### 注意力机制

对于注意力机制，我们只需要对模型进行简单修改即可完成适配。最新版本的 transformers 由于开放了 ``AttentionInterface``，因此对于支持 ``AttentionInterface`` 的模型，我们可以使用更简单的 register 方式来实现注意力机制适配（该功能还在实验中）。接下来只介绍一下通用的注意力机制修改方法。

其实相对比较简单，``PrefixGrouper`` 已经封装好了对应的注意力操作，我们只需要在模型对应的注意力操作的位置进行少量代码修改：

```py
if prefix_grouper is None:
    # 这里是原始 attention 接口，也就是 baseline 方法，我们只需要不传入 prefix_grouper 参数即可
    attn_output = _flash_attention_forward(...)
else:
    # ===== PrefixGrouper Start =====
    def attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor, *args, **kwargs):
        # 这里是一个适配函数，因为 prefix_grouper.forward 调用 attention 接口的时候采用固定的参数，不做处理，有时候会有 q、k、v 维度的顺序需要修改一下的问题，或者新增一些其他参数（如根据输入的 q 来输入 q_len=q.size(2)），都可以在这里完成。
        return _flash_attention_forward(...)
    
    # 传入对应的参数进行 forward 即可（详见使用文档）
    attn_output = prefix_grouper.forward(...)
    # ====== PrefixGrouper End ======
```

然后，我们只需要对模型的 forward 过程从外到内增加 prefix_grouper 参数，使其能够层层传递进来即可。

### 位置编码

对于位置编码，拼接之后输入的 responses 应该共用同一个起始 id，也就是 ``[0, 1, ..., n（前缀长度）, n + 1, ..., m（后缀1）, n + 1, ..., m2（后缀2）, n + 1, ..., m3（后缀3）, ...（依此类推）]``。一些有用的信息：``prefix_grouper.group_info[i].prefix_len`` 和 ``prefix_grouper.group_info[i].suffix_lens`` 可以获得第 ``i`` 个 sample 的前缀/后缀长度信息（不算 padding 的有效 token 数）；``prefix_grouper.padding_mask`` 可以获得前缀和后缀 concat 之后的 input tensor 的 attention mask。上述信息可以用于辅助 position ids 的计算。

对于``快速上手``部分的模型，我们已经对位置编码进行了适配，可以在对应代码中查看具体的示例。

### 开始训练

我们在 ``tests`` 中提供了一些完整的例子，其中模拟了 GRPO 的训练流程，作为参考。

## 使用文档

在本节我们将介绍最核心的 API 接口的文档。

### PrefixGrouper

#### PrefixGrouper(group_info: Optional[List[List[int]]] = None, device=None, padding_mode: Union[str, torch.Tensor] = "right")

参数 ``group_info`` 为 List[List[str]]，外层列表代表 sample 数量，内层列表代表 sample prefix 和 response suffix 的实际长度，其中内层列表的第一个元素代表 prefix 长度，剩余元素代表各个 response suffix 的长度。该参数可以为 ``None``，此时需要在之后手动调用 ``init``（与 ``PrefixGrouper.__init__`` 签名相同）来实现延迟初始化。

``device``：PrefixGrouper 初始化流程所使用的 device，注意这并不代表注意力操作的实际 device，实际 forward 时会根据输入的 tensor device 来自动转换。

``padding_mode``：类型为 ``str`` 时，可以为 "left" 或 "right"，代表着左 padding 还是右 padding（这里需要输入的 token 是稠密的，只在两边存在 padding）；类型为 ``torch.Tensor`` 时，是一个 padding mask，与 attention mask 类似，shape 为 [b, seq_len]，``True`` 元素代表非 padding 位置。

在使用 ``PrefixGrouper`` 时，有以下常见用法：

- 使用 concat_input 处理输入（推荐）

这时，``padding_mode`` 推荐为 "left" 或 "right"。

```py
prefix_grouper = PrefixGrouper(group_info, padding_mode="right")  # padding_mode="left" 也可
```

- 自定义处理输入（即前缀与后缀的 concat 流程由你自行实现）

这时，我们也支持输入 padding_mask（用于指示哪些位置是非 padding token）：

```py
prefix_grouper = PrefixGrouper(group_info, padding_mode=padding_mask)
```

#### PrefixGrouper.concat_input(self, prefix: torch.Tensor, prefix_mask: torch.Tensor, suffix: torch.Tensor, suffix_mask: torch.Tensor)

将 ``prefix`` 与 ``suffix`` 按照 ``group_info`` 来进行 concat，需要传入 ``prefix_mask`` 与 ``suffix_mask`` 来指定哪些位置是非 padding token。

``prefix`` 与 ``suffix`` 的 shape 可为 ``[b, seq_len]`` 或 ``[b, seq_len, dim]``，其中 ``prefix.shape[0]`` 应当为 sample 数量，``suffix.shape[0]`` 应当为全部 response 的数量。

``prefix_mask`` 与 ``suffix_mask`` 的 shape 应为 ``[b, seq_len]``。

#### PrefixGrouper.forward(self, __attn_func: AttnFuncType, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs)

根据传入的 ``__attn_func`` 和一系列参数进行 attention 操作。``__attn_func`` 应当按照以下顺序接受参数：``q``、``k``、``v``、``attn_mask``、``*args``、``**kwargs``。如果已有的 attention 函数参数与上述顺序有差异，可以自定义一个 adapter 函数，按照上述顺序接收参数，然后转换成 attention 函数需要的参数形式进行调用。注意，``PrefixGrouper`` 会自动计算需要的 attention mask，请不要在 ``PrefixGrouper.forward`` 中手动传入 attention mask 参数。

另外，为了统一，``PrefixGrouper.forward`` 接受的 ``q``、``k``、``v`` shape 应当为 ``[b, num_heads, seq_len, head_dim]``，同样地，调用 ``__attn_func`` 时传入的 ``q``、``k``、``v`` shape 也是 ``[b, num_heads, seq_len, head_dim]``。``__attn_func`` 所返回的输出 shape 应当为 ``[b, seq_len, num_heads, head_dim]``，``PrefixGrouper.forward`` 所返回的输出 shape 也为 ``[b, seq_len, num_heads, head_dim]``。在编写适配代码时请注意输入输出的维度顺序，在需要的位置进行 transpose。

#### PrefixGrouper.split_output(self, output: torch.Tensor, include_prefix_last: int = 0)

``output``：shape 为 ``[b, seq_len, dim]``

``include_prefix_last``：将前缀最后的 n 个 token 转换为共享的后缀 token，为 0 时则代表不转换（详见使用教程的数据输入输出部分）。

## 未来计划

- [ ] Hugging Face transformers ``AttentionInterface`` 集成（该功能正在测试中）
- [ ] 其他训练设备测试（``NPU`` 正在测试中，目前暂未发现不兼容问题）
- [ ] 其他模型的测试用例（我们计划 release 纯文本版本的 ``Qwen2.5``、``Qwen3`` 模型的测试用例）
- [ ] 兼容其他注意力实现（``EagerAttention``、``SDPA``）

## 数据使用声明

本项目使用的测试数据仅用于**学术研究目的**，具有以下严格限制：

1. **禁止任何形式的商业用途**

2. **禁止数据重新分发**

3. **禁止尝试去匿名化操作**

## 引用

如果您认为这项工作有帮助，可以引用以下论文：

```bibtex
@misc{liu2025prefixgrouperefficientgrpo,
      title={Prefix Grouper: Efficient GRPO Training through Shared-Prefix Forward}, 
      author={Zikang Liu and Tongtian Yue and Yepeng Tang and Longteng Guo and Junxian Cai and Qingbin Liu and Xi Chen and Jing Liu},
      year={2025},
      eprint={2506.05433},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.05433}, 
}
```
