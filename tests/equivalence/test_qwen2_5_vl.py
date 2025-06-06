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
        padding_side="left",
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
        name: param.grad.clone().cpu()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def test_baseline(
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
    resp_ids = resp_inputs["input_ids"].to(device)
    resp_mask = resp_inputs["attention_mask"].to(device)
    logits = res.logits[:, prompt_ids.shape[1] - 1: -1].float()
    loss = (logits.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)).exp()
    loss = loss * resp_mask
    loss = (loss.sum(-1) / resp_mask.sum(-1)).mean()
    if loss.requires_grad:
        (-loss).backward()
    return (
        [output[mask.bool()] for output, mask in zip(logits, resp_mask)],
        get_grad(model)
    )


def test_prefix_grouper(
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
    suffix_inputs = processor(
        text=[r for resps in responses for r in resps],
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=False,
    )
    suffix_ids = suffix_inputs["input_ids"].to(device)
    suffix_mask = suffix_inputs["attention_mask"].to(device)
    suffix_lens = suffix_mask.sum(-1)
    lengths = [len(resps) for resps in responses]
    suffix_lens = [
        [int(t.item()) for t in chunk]
        for chunk in torch.split(suffix_lens, lengths, dim=0)
    ]
    group_info = [[p_len, *s_lens] for p_len, s_lens in zip(prefix_lens, suffix_lens)]
    prefix_grouper = PrefixGrouper(
        group_info=group_info,
        padding_mode="right",
        device=device,
    )
    prompt_inputs["input_ids"] = prefix_grouper.concat_input(prompt_ids, prompt_mask, suffix_ids, suffix_mask)
    prompt_inputs["attention_mask"] = prefix_grouper.padding_mask
    res = model(**prompt_inputs, use_cache=False, prefix_grouper=prefix_grouper)
    # Calculate loss and backward
    prefix_output, prefix_mask, suffix_output, suffix_mask = (
        prefix_grouper.split_output(res.logits)
    )
    suffix_ids = prefix_grouper.convert_padding(suffix_ids, suffix_mask, padding_mode="right")
    suffix_output = suffix_output[:, :-1].float()
    suffix_mask = suffix_mask[:, 1:]
    loss = (suffix_output.gather(-1, suffix_ids.unsqueeze(-1)[:, 1:]).squeeze(-1) - suffix_output.logsumexp(-1)).exp()
    loss = loss * suffix_mask
    loss = (loss.sum(-1) / suffix_mask.sum(-1)).mean()
    if loss.requires_grad:
        (-loss).backward()
    return (
        [out[mask] for out, mask in zip(suffix_output, suffix_mask)],
        get_grad(model),
    )


def test_prefix_grouper_include_last(
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
    suffix_inputs = processor(
        text=[r for resps in responses for r in resps],
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=False,
    )
    suffix_ids = suffix_inputs["input_ids"].to(device)
    suffix_mask = suffix_inputs["attention_mask"].to(device)
    suffix_lens = suffix_mask.sum(-1)
    lengths = [len(resps) for resps in responses]
    suffix_lens = [
        [int(t.item()) for t in chunk]
        for chunk in torch.split(suffix_lens, lengths, dim=0)
    ]
    group_info = [[p_len, *s_lens] for p_len, s_lens in zip(prefix_lens, suffix_lens)]
    prefix_grouper = PrefixGrouper(
        group_info=group_info,
        padding_mode="right",
        device=device,
    )
    prompt_inputs["input_ids"] = prefix_grouper.concat_input(prompt_ids, prompt_mask, suffix_ids, suffix_mask)
    prompt_inputs["attention_mask"] = prefix_grouper.padding_mask
    res = model(**prompt_inputs, use_cache=False, prefix_grouper=prefix_grouper)
    # Calculate loss and backward
    # NOTE: The last token of the prefix should be changed to the first input token of the suffix
    # NOTE: The new ``suffix_mask`` will include the last prefix token at the start
    prefix_output, prefix_mask, suffix_output, suffix_mask_out = (
        prefix_grouper.split_output(res.logits, include_prefix_last=1)
    )
    suffix_ids = prefix_grouper.convert_padding(suffix_ids, suffix_mask, padding_mode="right")
    suffix_output = suffix_output[:, :-1].float()
    suffix_mask_out = suffix_mask_out[:, 1:]
    loss = (suffix_output.gather(-1, suffix_ids.unsqueeze(-1)).squeeze(-1) - suffix_output.logsumexp(-1)).exp()
    loss = loss * suffix_mask_out
    loss = (loss.sum(-1) / suffix_mask_out.sum(-1)).mean()
    if loss.requires_grad:
        (-loss).backward()
    return (
        [out[mask] for out, mask in zip(suffix_output, suffix_mask_out)],
        get_grad(model),
    )


def test_prefix_grouper_include_last_auto_group_info(
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

    # NOTE: Suppose we have got the final responses by the model here
    suffix_inputs = processor(
        text=[r for resps in responses for r in resps],
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=False,
    )
    suffix_ids = suffix_inputs["input_ids"].to(device)
    suffix_mask = suffix_inputs["attention_mask"].to(device)
    prefix_grouper = PrefixGrouper.from_ungrouped_masks(
        # NOTE: The ``group_info`` can be automatically calculated through masks!
        prefix_mask=prompt_mask,
        suffix_mask=suffix_mask,
        group_sizes=[len(resps) for resps in responses],
        device=device,
        padding_mode="right",
    )
    prompt_inputs["input_ids"] = prefix_grouper.concat_input(prompt_ids, prompt_mask, suffix_ids, suffix_mask)
    prompt_inputs["attention_mask"] = prefix_grouper.padding_mask
    res = model(**prompt_inputs, use_cache=False, prefix_grouper=prefix_grouper)
    # Calculate loss and backward
    # NOTE: The last token of the prefix should be changed to the first input token of the suffix
    # NOTE: The new ``suffix_mask`` will include the last prefix token at the start
    prefix_output, prefix_mask, suffix_output, suffix_mask_out = (
        prefix_grouper.split_output(res.logits, include_prefix_last=1)
    )
    suffix_ids = prefix_grouper.convert_padding(suffix_ids, suffix_mask, padding_mode="right")
    suffix_output = suffix_output[:, :-1].float()
    suffix_mask_out = suffix_mask_out[:, 1:]
    loss = (suffix_output.gather(-1, suffix_ids.unsqueeze(-1)).squeeze(-1) - suffix_output.logsumexp(-1)).exp()
    loss = loss * suffix_mask_out
    loss = (loss.sum(-1) / suffix_mask_out.sum(-1)).mean()
    if loss.requires_grad:
        (-loss).backward()
    return (
        [out[mask] for out, mask in zip(suffix_output, suffix_mask_out)],
        get_grad(model),
    )


def load_input_data(model_path, processor, device, data: list):
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
    return forward_kwargs


def test(model_path: str, empty_cache_func: Callable[[], Any], device=None):
    processor = AutoProcessor.from_pretrained(model_path)
    data = ["video1", "video2"]
    # Run forward and backward for compare
    # NOTE: The three methods use the same ``load_input_data`` function, which means this part is totally 
    # the same among these methods, and all we should focus on is the ``forward_backward_xxx`` function.
    # Baseline method
    outputs, grads = test_baseline(**load_input_data(model_path, processor, device, data))
    empty_cache_func()
    
    # # PrefixGrouper with separate last prefix token (legacy version)
    # outputs2, grads2 = test_prefix_grouper(**load_input_data(model_path, processor, device, data))
    # empty_cache_func()
    # # PrefixGrouper with shared last prefix token (legacy version)
    # outputs3, grads3 = test_prefix_grouper_include_last(**load_input_data(model_path, processor, device, data))
    # empty_cache_func()
    
    # PrefixGrouper best practice for now! Simplest usage.
    outputs4, grads4 = test_prefix_grouper_include_last_auto_group_info(**load_input_data(model_path, processor, device, data))
    empty_cache_func()
    breakpoint()


def test_cuda(model_path: str):
    test(model_path, torch.cuda.empty_cache, "cuda")


def test_npu(model_path: str):
    test(model_path, torch.npu.empty_cache, "npu")


if __name__ == "__main__":
    test_cuda("your_model_path_here")
