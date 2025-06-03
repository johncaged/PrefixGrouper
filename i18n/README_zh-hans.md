# PrefixGrouper

<h4 align="center">
    <p>
        <a href="../README.md">English</a> |
        <b>简体中文</b>
    </p>
</h4>

``PrefixGrouper`` 是一个即插即用的高效 GRPO 训练工具，只需在已有的训练代码库上进行少量的修改，即可实现计算量减少、显存占用降低、训练加速。此外，该工具还可以用到除 GRPO 之外的其他需要共享前缀训练 / 推理的场景。

目前主流的 GRPO 训练流程中，策略模型的训练主要是通过将前缀（通常是问题、多模态输入等）复制 ``G`` 次来实现的，那么显而易见的，当训练数据的前缀足够长时（如长上下文推理、图像或长视频推理等），训练过程中产生的冗余计算就变得不可忽视，从而导致显存占用增加、计算量增大、训练速度下降。对此，我们提出了 ``PrefixGrouper``，一个即插即用的 GRPO 训练工具，通过共享前缀的 forward 来实现高效训练。相对地，显存占用的降低反过来可以使得相同数量的显卡可以支持的最大 group_size 变大，这对于 GRPO 算法来说是至关重要的。

## 最新动态

- [2025/6/3] 我们正式发布 ``PrefixGrouper`` 工具。技术报告即将推出，敬请期待。

## 安装

```py
pip install prefix_grouper
```

## 快速上手

为了让 ``PrefixGrouper`` 更简单易用，我们提供了部分模型的修改示例。

- 模型文件修改示例请查看 ``examples``。为了更清晰，我们在关键修改部分采用 "PrefixGrouper Start" 和 "PrefixGrouper End" 注释包裹。
- 模拟整体训练流程的示例请查看 ``tests/equivalence``。我们提供一个训练 step 的几乎完整的流程。

如果你恰好需要使用示例中的模型进行训练，那么可以直接将示例中的代码引入到你的代码库。然而，我们还是建议你简单了解一下 ``PrefixGrouper`` 的使用教程，以更清楚地了解该工具的运行流程。

## 使用教程

简单来说，``PrefixGrouper`` 主要需要对 GRPO 训练流程进行三个方面的修改：数据输入输出、注意力机制和位置编码。在全文中，我们将一个问题 query（也就是前缀）所对应的数据称为一个 sample，而由模型根据前缀所采样生成的每一个输出，我们称为一个 response。

### 数据输入输出

为了降低前缀 forward 冗余、尽可能利用并行加速，``PrefixGrouper`` 首先将 batch 内的每一个 sample 与它所对应的多个 responses 进行拼接，得到 grouped 输入（以下进行伪代码示例）：

```py
# 假设这里我们已经拿到了前缀的 input_ids，它是一个已经 padding 过的 torch.Tensor，shape 为 [b, seq_len1]（其对应的 mask 也是同样的 shape）
prompt_ids = ...
prompt_mask = ...

# 这里我们假设已经获取了由模型生成的 responses，生成的方式（无论是 model.generate 还是 vLLM 等）我们在这里不关心，只要最终得到 responses 的输出 str 或者 input_ids 即可。
# 这里我们将 responses 设置成 List[List[str]]，最外层的 list 代表 sample 的数量，也就是 prompt_ids shape 对应的 b，最内层的 list 则代表 response 的数量，也就是每一个 sample 产生了多少个 response。注意这里每个 sample 对应的 response 数量可以不同，PrefixGrouper 对此进行了适配。
responses: List[List[str]] = ...
# 将 responses 铺平，使用 tokenizer、processor 等将其 tokenize 并 padding 成 input_ids。此处的 shape 应该为 [b * group_size（假设每个 sample 的 response 数量都是 group_size）, seq_len2]。
suffix_ids = ...
suffix_mask = ...

# ====== 至此，所需要的输入就处理完毕了 ======

# 接下来，我们需要创建一个 PrefixGrouper 实例，用于共享前缀输入
# 首先，PrefixGrouper 需要一个必要的信息，也就是 group_info，它需要知道我们刚刚拼接的输入中哪些部分是前缀、哪些部分是后缀。
# 同样的，group_info 是一个嵌套 list，最外层代表 sample 数量 b，最内层则代表前缀、后缀的长度（其中第一个元素就是前缀长度，从第二个元素开始就代表着每一个后缀的长度），需要与我们刚刚处理的输入对应。举例：group_info[0][0]，代表第一个 sample 的前缀长度，group_info[0][1]，则代表第一个 sample 的第一个后缀（response）长度。这里的长度指的是有效长度，也就是“实际不为 padding 的，包含了多模态 token 数量的长度”。
# 我们在这里提供一些思路：对于 Qwen2.5-VL 来说，由 processor 处理过的 inputs，其实本身是包含了对应视觉 token 数量的占位符的，因此我们只需要计算 prompt_mask.sum(-1)，就可以得到每一个前缀的真实长度；对于 LLaVA-NeXT 系列的 codebase 来说，我们可以修改一下数据的处理，首先将 prompt_ids 输入到 prepare_inputs_labels_for_multimodal，让其进行多模态 token 的拼接，然后得到的结果可以通过类似的 attention_mask.sum(-1) 来获取真实的前缀长度。至于后缀长度，可以直接通过后缀的 suffix_mask.sum(-1) 来得到。
group_info: List[List[int]] = ...
# 初始化一个 PrefixGrouper 实例。
prefix_grouper = PrefixGrouper(
    group_info=group_info,
    padding_mode="right",
    device=device,
)
# 这里我们利用 PrefixGrouper 将分散的输入拼接成最终的 input_ids，其 shape 为 [b, seq_len]。注意 seq_len 可能不等于 seq_len1 + seq_len2 * group_size，因为拼接过程会对输入进行优化，去掉冗余的 padding。
input_ids = prefix_grouper.concat_input(prompt_ids, prompt_mask, suffix_ids, suffix_mask)
# 这里进行模型 forward，只需要多加一个参数即可
res = model(*args, **kwargs, prefix_grouper=prefix_grouper)
# ====== 至此，forward 流程完毕 ======

# 接下来我们开始计算 loss，并进行 backward
# 由于训练过程中采用自回归进行建模，因此我们需要注意一个细节，就是模型的前缀和后缀之间的边界位置的 token。我们知道，模型开始输出第一个 token 的时机是“前缀最后一个 token 的输入”，那么刚刚我们拼接输入的时候，每一个 response 产生第一个 token 的位置应当是前缀部分的最后一个 token 输入的位置。在这里 PrefixGrouper 给出了两种实现方式：
# 1. 前缀最后一个 token 还保留在前缀，那么在处理输出的时候，应当使用 include_prefix_last=1，这样每一个 suffix_output 都会拼接其对应的前缀最后一个 token 在最前面（也就是接下来的使用示例）。
# 2. 前缀最后一个 token 在最开始的时候就复制，然后与每一个后缀拼接到一起，这样就不需要附带 include_prefix_last 参数，模型自动根据原来的 group_info 进行 split 即可（稍微麻烦一点）。
# 举例（1）：
# 前缀：<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n
# 后缀：Hello, can I help you?<|im_end|>\n
# 在这里，模型 response 的第一个 token（也就是后缀 tokenize 得到的第一个 input_id）是由前缀的最后一个 token（也就是 \n）输出的，当我们有多个后缀时，每一个后缀会共用前缀最后一个 token 的位置进行 loss 计算，那么此时我们就应该使用 include_prefix_last=1
# 举例（2）（稍微麻烦一点）：
# 前缀：<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant
# 后缀：\nHello, can I help you?<|im_end|>\n
# 在这里，如果有多个 response 的话，我们需要手动在每一个后缀的前面拼接前缀的最后一个 token（也就是 \n），那么前缀也就相应地去掉了最后一个 token。这时，我们就不需要附带 include_prefix_last 参数（也就是 include_prefix_last=0）
# 得到的最终输出：prefix_output、prefix_mask、suffix_output、suffix_mask，其中 prefix_output 和 prefix_mask 的 shape 为 [b, seq_len1]，对应着 sample 数量；suffix_output 和 suffix_mask 的 shape 为 [b * group_size, seq_len2]，对应着 response 的数量。对应的 mask 则代表了有效 token 的位置（也就是非 padding 的位置）。
prefix_output, prefix_mask, suffix_output, suffix_mask = (
    prefix_grouper.split_output(res.logits, include_prefix_last=1)
)
# ====== 至此，输入输出的流程全部完成 ======

# 获取了正常输出之后，接下来就可以按照 GRPO 来计算 loss、反向传播了，与标准 GRPO 流程完全一致。
suffix_output = suffix_output[:, :-1]
suffix_mask = suffix_mask[:, 1:]
loss = (suffix_output.gather(-1, suffix_ids.unsqueeze(-1)).squeeze(-1) - suffix_output.logsumexp(-1)).exp()
loss = loss * suffix_mask
loss = (loss.sum(-1) / suffix_mask.sum(-1)).mean()
(-loss).backward()
```

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

对于位置编码，拼接之后输入的 responses 应该共用同一个起始 id，也就是 ``[0, 1, ..., n（前缀长度）, n + 1, ..., m（后缀1）, n + 1, ..., m2（后缀2）, n + 1, ..., m3（后缀3）, ...（依此类推）]``。

对于``快速上手``部分的模型，我们已经对位置编码进行了适配，可以在对应代码中查看具体的示例。

### 开始训练

我们在 ``tests`` 中提供了一些完整的例子，其中模拟了 GRPO 的训练流程，作为参考。

## 使用文档

在本节我们将介绍最核心的 API 接口的文档。

### PrefixGrouper

#### PrefixGrouper(group_info: List[List[int]], device=None, padding_mode: Union[str, torch.Tensor] = "right")

参数 ``group_info`` 为 List[List[str]]，外层列表代表 sample 数量，内层列表代表 sample prefix 和 response suffix 的实际长度，其中内层列表的第一个元素代表 prefix 长度，剩余元素代表各个 response suffix 的长度。

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

``prefix`` 与 ``suffix`` 的 shape 可为 [b, seq_len] 或 [b, seq_len, dim]，其中 ``prefix.shape[0]`` 应当为 sample 数量，``suffix.shape[0]`` 应当为全部 response 的数量。

``prefix_mask`` 与 ``suffix_mask`` 的 shape 应为 [b, seq_len]。

#### PrefixGrouper.forward(self, __attn_func: AttnFuncType, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs)

根据传入的 ``__attn_func`` 和一系列参数进行 attention 操作。``__attn_func`` 应当按照以下顺序接受参数：``q``、``k``、``v``、``attn_mask``、``*args``、``**kwargs``。如果已有的 attention 函数参数与上述顺序有差异，可以自定义一个 adapter 函数，按照上述顺序接收参数，然后转换成 attention 函数需要的参数形式进行调用。注意，``PrefixGrouper`` 会自动计算需要的 attention mask，请不要在 ``PrefixGrouper.forward`` 中手动传入 attention mask 参数。

另外，为了统一，``PrefixGrouper.forward`` 接受的 ``q``、``k``、``v`` shape 应当为 [b, num_heads, seq_len, head_dim]，同样地，调用 ``__attn_func`` 时传入的 ``q``、``k``、``v`` shape 也是 [b, num_heads, seq_len, head_dim]。``__attn_func`` 所返回的输出 shape 应当为 [b, seq_len, num_heads, head_dim]，``PrefixGrouper.forward`` 所返回的输出 shape 也为 [b, seq_len, num_heads, head_dim]。在编写适配代码时请注意输入输出的维度顺序，在需要的位置进行 transpose。

#### PrefixGrouper.split_output(self, output: torch.Tensor, include_prefix_last: int = 0)

``output``：shape 为 [b, seq_len, dim]
``include_prefix_last``：将前缀最后的 n 个 token 转换为共享的后缀 token，为 0 时则代表不转换（详见使用教程的数据输入输出部分）。

## 数据使用声明

本项目使用的测试数据仅用于**学术研究目的**，具有以下严格限制：

1. **禁止任何形式的商业用途**

2. **禁止数据重新分发**

3. **禁止尝试去匿名化操作**

## 引用

[TODO]
