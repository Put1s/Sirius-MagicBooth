import random

import torch
from diffusers import DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline
from torch import autocast

from logs import get_train_bot_logger, log_function

logger = get_train_bot_logger()

height = 512
width = 512


def load_model(model_path):
    pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.unet.to(memory_format=torch.channels_last)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    pipeline = pipeline.to('cuda')

    return pipeline


@log_function(logger)
def generate_images(cls, prompts: list, negative_prompts: list, model_path, seeds, n=3):
    pipeline = load_model(model_path)
    for prm, negative_prompt, seed in zip(prompts, negative_prompts, seeds):
        temp_prompt: str = random.choice(prm['prompts'])
        prompt = temp_prompt.format(cls)
        yield func(prompt, negative_prompt, pipeline, seed, n)


@autocast('cuda')
@torch.inference_mode()
def func(prompt, negative_prompt, pipeline, seed, n):
    g_cuda = torch.Generator(device="cuda")
    g_cuda.manual_seed(seed)
    images = pipeline(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=n,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=g_cuda,
    ).images

    return images
