import os
import json
import importlib.util
from typing import List, Callable, Any
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from prefix_grouper import PrefixGrouper


def load_module(name: str, path: str):
    """
    Load module based on the python file path.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_prefix_inputs(processor, video_file: str, instruction: str):
    """
    Prepare prefix input (following the standard example of Qwen2.5-VL)
    """
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_file,
                    "min_pixels": 100352,
                    "max_pixels": 100352,
                    "max_frames": 16,
                    "fps": 1,
                },
                {
                    "type": "text",
                    "text": instruction,
                },
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        message, return_video_kwargs=True
    )
    return prompt, video_inputs, video_kwargs


def process_inputs(processor, videos, instructions, device):
    """
    Process the inputs (following the standard example of Qwen2.5-VL)
    """
    prompt_inputs = processor(
        text=instructions,
        images=None,
        videos=videos,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=False,
        fps=1.0,
    )
    transfer_device_keys = [
        "input_ids",
        "attention_mask",
        "pixel_values_videos",
        "video_grid_thw",
    ]
    for key in transfer_device_keys:
        prompt_inputs[key] = prompt_inputs[key].to(device)
    return prompt_inputs


def batch_repeat_interleave(lst, repeats):
    """
    Simplified version of PyTorch's repeat_interleave for Python lists
    Only repeats elements at the outermost level
    """
    if isinstance(repeats, int):
        return [item for item in lst for _ in range(repeats)]

    if len(repeats) != len(lst):
        raise ValueError("Length of repeats must match length of input list")

    return [item for i, item in enumerate(lst) for _ in range(repeats[i])]


def get_grad(model):
    return {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def forward_backward_base(
    model_path: str,
    processor,
    videos: List,
    instructions: List[str],
    responses: List[List[str]],
    load_kwargs: dict,
    device,
):
    """
    Run forward and backward pass in baseline.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL,
    )

    model = Qwen2_5_VL.from_pretrained(model_path, **load_kwargs, device_map=device)
    model.gradient_checkpointing_enable()
    # NOTE: Suppose we have got the final responses by the model here
    response_sizes = [len(resp) for resp in responses]
    resp_inputs = processor(
        text=[r for resp in responses for r in resp],
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=False,
    )
    # Process the inputs here.
    # NOTE: We should repeat the prefix inputs here.
    prompt_inputs = process_inputs(
        processor,
        batch_repeat_interleave(videos, response_sizes),
        batch_repeat_interleave(instructions, response_sizes),
        device,
    )
    prompt_ids = prompt_inputs.pop("input_ids")
    prompt_mask = prompt_inputs.pop("attention_mask")
    prompt_inputs["input_ids"] = torch.cat(
        [
            prompt_ids,
            resp_inputs["input_ids"].to(device),
        ],
        dim=1,
    )
    prompt_inputs["attention_mask"] = torch.cat(
        [
            prompt_mask,
            resp_inputs["attention_mask"].to(device),
        ],
        dim=1,
    )
    res = model(**prompt_inputs, use_cache=False)
    # Calculate loss and backward
    resp_ids = resp_inputs["input_ids"]
    resp_mask = resp_inputs["attention_mask"]
    prompt_mask_cumsum = prompt_mask.cumsum(dim=1)
    prompt_mask = (prompt_mask_cumsum == prompt_mask_cumsum.max(dim=1).values.unsqueeze(1)) & prompt_mask
    output_mask = torch.cat([prompt_mask, resp_mask.to(device)], dim=1).bool()
    loss_mask = output_mask[:, :-1]
    logits = res.logits[:, :-1]
    targets = prompt_inputs["input_ids"][:, 1:]
    loss = (logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)).exp()
    loss = loss * loss_mask
    loss = (loss.sum(-1) / loss_mask.sum(-1)).mean()
    (-loss).backward()
    return (
        [output[mask] for output, mask in zip(res.logits, output_mask)],
        get_grad(model)
    )


def forward_backward_prefix_grouper(
    model_path: str,
    processor,
    videos: List,
    instructions: List[str],
    responses: List[List[str]],
    load_kwargs: dict,
    device,
):
    """
    Run forward and backward pass in PrefixGrouper.
    """
    # NOTE: We import the module using python file path
    modeling_qwen2_5_vl = load_module(
        "modeling_qwen2_5_vl", "./examples/qwen2_5_vl/modeling_qwen2_5_vl.py"
    )
    Qwen2_5_VLPrefixGrouper = modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLPrefixGrouper.from_pretrained(
        model_path, **load_kwargs, device_map=device
    )
    model.gradient_checkpointing_enable()
    # NOTE: The last token of the prefix should be changed to the first input token of the suffix
    # NOTE: If you change other models using a different chat template, then pay attention that the
    # token may not be one char, but it may be multi-chars such as "\n\n". But in Qwen2.5-VL, it
    # happens to be one char "\n".
    suffix_start_str = instructions[0][-1]
    instructions = [instruct[:-1] for instruct in instructions]
    prompt_inputs = process_inputs(processor, videos, instructions, device)
    prompt_ids = prompt_inputs.pop("input_ids")
    prompt_mask = prompt_inputs.pop("attention_mask")

    prefix_lens = prompt_mask.sum(dim=1).tolist()
    # NOTE: Suppose we have got the final responses by the model here
    responses = [[suffix_start_str + r for r in resps] for resps in responses]
    resp_ids_list = [
        [
            torch.tensor(
                processor.tokenizer(r).input_ids,
                dtype=torch.long,
                device=device,
            )
            for r in resps
        ]
        for resps in responses
    ]
    group_info = [
        [p_len, *(r.shape[0] for r in r_list)]
        for p_len, r_list in zip(prefix_lens, resp_ids_list)
    ]
    # NOTE: We should concat the input ids manually rather than concat the str first then
    # use processor, because concatenating different suffixes head to tail will cause
    # tokenize ids to merge
    final_input_ids = [
        torch.cat([prompt_ids_, *r_list])
        for prompt_ids_, r_list in zip(prompt_ids, resp_ids_list)
    ]
    prompt_inputs["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        final_input_ids,
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    prompt_inputs["attention_mask"] = (
        prompt_inputs["input_ids"] != processor.tokenizer.pad_token_id
    )
    prefix_grouper = PrefixGrouper(
        group_info=group_info,
        padding_mode=prompt_inputs["attention_mask"],
        device=device,
    )
    res = model(**prompt_inputs, use_cache=False, prefix_grouper=prefix_grouper)
    # Calculate loss and backward
    prefix_output, prefix_mask, suffix_output, suffix_mask = (
        prefix_grouper.split_output(res.logits)
    )
    resp_ids_list = [r for r_list in resp_ids_list for r in r_list]
    loss_list = []
    for resp_ids, output, mask in zip(resp_ids_list, suffix_output, suffix_mask):
        output = output[mask][:-1]
        logps = output.gather(-1, resp_ids.unsqueeze(-1)[1:]).squeeze(-1) - output.logsumexp(-1)
        loss_list.append(logps.exp().mean())
    loss = -torch.stack(loss_list).mean()
    loss.backward()

    return (
        [out[mask] for out, mask in zip(suffix_output, suffix_mask)],
        get_grad(model),
    )


def forward_backward_prefix_grouper_include_last(
    model_path: str,
    processor,
    videos: List,
    instructions: List[str],
    responses: List[List[str]],
    load_kwargs: dict,
    device,
):
    """
    Run forward and backward pass in PrefixGrouper.
    """
    # NOTE: We import the module using python file path
    modeling_qwen2_5_vl = load_module(
        "modeling_qwen2_5_vl", "./examples/qwen2_5_vl/modeling_qwen2_5_vl.py"
    )
    Qwen2_5_VLPrefixGrouper = modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLPrefixGrouper.from_pretrained(
        model_path, **load_kwargs, device_map=device
    )
    model.gradient_checkpointing_enable()
    prompt_inputs = process_inputs(processor, videos, instructions, device)
    prompt_ids = prompt_inputs.pop("input_ids")
    prompt_mask = prompt_inputs.pop("attention_mask")

    prefix_lens = prompt_mask.sum(dim=1).tolist()
    # NOTE: Suppose we have got the final responses by the model here
    resp_ids_list = [
        [
            torch.tensor(
                processor.tokenizer(r).input_ids,
                dtype=torch.long,
                device=device,
            )
            for r in resps
        ]
        for resps in responses
    ]
    group_info = [
        [p_len, *(r.shape[0] for r in r_list)]
        for p_len, r_list in zip(prefix_lens, resp_ids_list)
    ]
    # NOTE: We should concat the input ids manually rather than concat the str first then
    # use processor, because concatenating different suffixes head to tail will cause
    # tokenize ids to merge
    final_input_ids = [
        torch.cat([prompt_ids_, *r_list])
        for prompt_ids_, r_list in zip(prompt_ids, resp_ids_list)
    ]
    prompt_inputs["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        final_input_ids,
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    prompt_inputs["attention_mask"] = (
        prompt_inputs["input_ids"] != processor.tokenizer.pad_token_id
    )
    prefix_grouper = PrefixGrouper(
        group_info=group_info,
        padding_mode=prompt_inputs["attention_mask"],
        device=device,
    )
    res = model(**prompt_inputs, use_cache=False, prefix_grouper=prefix_grouper)
    # Calculate loss and backward
    # NOTE: The last token of the prefix should be changed to the first input token of the suffix
    prefix_output, prefix_mask, suffix_output, suffix_mask = (
        prefix_grouper.split_output(res.logits, include_prefix_last=1)
    )
    resp_ids_list = [r for r_list in resp_ids_list for r in r_list]
    loss_list = []
    for resp_ids, output, mask in zip(resp_ids_list, suffix_output, suffix_mask):
        output = output[mask][:-1]
        logps = output.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1) - output.logsumexp(-1)
        loss_list.append(logps.exp().mean())
    loss = -torch.stack(loss_list).mean()
    loss.backward()

    return (
        [out[mask] for out, mask in zip(suffix_output, suffix_mask)],
        get_grad(model),
    )


def test(model_path: str, empty_cache_func: Callable[[], Any], device=None):
    # -------------- Shared Code between baseline and PrefixGrouper --------------
    processor = AutoProcessor.from_pretrained(model_path)
    # Load data here
    data = ["video1", "video2"]
    video_base_dir = "./tests/test_inputs/videos"
    text_base_dir = "./tests/test_inputs/captions"
    videos = []
    instructions = []
    responses = []
    for video_name in data:
        with open(os.path.join(text_base_dir, f"{video_name}.json")) as f:
            item = json.load(f)
        instruction = item["instruction"]
        new_instruction, video_inputs, video_kwargs = prepare_prefix_inputs(
            processor,
            os.path.join(video_base_dir, f"{video_name}.mp4"),
            instruction,
        )
        videos.extend(video_inputs)
        instructions.append(new_instruction)
        # NOTE: We concat the "<|im_end|>\n" here, because we get the raw str from the input file.
        responses.append([f"{resp}<|im_end|>\n" for resp in item["responses"]])
    # Prepare args
    forward_kwargs = dict(
        model_path=model_path,
        processor=processor,
        videos=videos,
        instructions=instructions,
        responses=responses,
        load_kwargs=dict(
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        ),
        device=device,
    )
    # ------------------------- Shared Code End ----------------------------------
    # Run forward and backward for compare
    outputs, grads = forward_backward_base(**forward_kwargs)
    empty_cache_func()
    outputs2, grads2 = forward_backward_prefix_grouper(**forward_kwargs)
    empty_cache_func()
    outputs3, grads3 = forward_backward_prefix_grouper_include_last(**forward_kwargs)
    empty_cache_func()


def test_cuda(model_path: str):
    test(model_path, torch.cuda.empty_cache, "cuda")


def test_npu(model_path: str):
    test(model_path, torch.npu.empty_cache, "npu")


if __name__ == "__main__":
    test_cuda("/public/jtma/weights/Qwen2.5-VL-3B-Instruct")
