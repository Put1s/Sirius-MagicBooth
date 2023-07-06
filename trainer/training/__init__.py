import json
import math

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel

from logs import get_train_logger
from logs import log_block
from logs import log_function
from trainer.training.train_epoch import train_one_epoch
from trainer.utils.CacheLatents import cache_latents
from trainer.utils.GenerateClassImages import generate_class_images

torch.backends.cudnn.benchmark = True

logger = get_train_logger()


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@log_function(logger, 'Initializing models')
def init_models(args):
    """Initialize models."""

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,  # "xyz man"
                "class_prompt": args.class_prompt,  # "man"
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir,
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    generate_class_images(args, accelerator)

    train_dataloader = cache_latents(args, accelerator)

    with log_block(logger, 'Creating models'):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            use_auth_token=True,
        )

        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            torch_dtype=torch.float32,
            use_auth_token=True,
        )

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            text_encoder.gradient_checkpointing_enable()

    optimizer_class = torch.optim.AdamW

    try:
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit
    except:
        pass

    # optimizers
    optimizer_unet = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_text_encoder = optimizer_class(
        text_encoder.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", use_auth_token=True
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    lr_scheduler_unet = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_unet,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    lr_scheduler_text_encoder = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_text_encoder,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_text_encoder,
    )
    (
        unet,
        text_encoder,
        optimizer_unet,
        optimizer_text_encoder,
        lr_scheduler_unet,
        lr_scheduler_text_encoder,
        train_dataloader,
    ) = accelerator.prepare(
        unet,
        text_encoder,
        optimizer_unet,
        optimizer_text_encoder,
        lr_scheduler_unet,
        lr_scheduler_text_encoder,
        train_dataloader,
    )

    return (
        accelerator,
        unet,
        text_encoder,
        optimizer_unet,
        optimizer_text_encoder,
        lr_scheduler_unet,
        lr_scheduler_text_encoder,
        train_dataloader,
        noise_scheduler
    )


@log_function(logger, 'Training models')
def train(
        accelerator,
        unet,
        text_encoder,
        train_dataloader,
        noise_scheduler,
        optimizer_unet,
        optimizer_text_encoder,
        lr_scheduler_text_encoder,
        lr_scheduler_unet,
        args,
        ticker,
):
    args.num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    progress_bar.set_description("Steps")

    # args.global_step = 0
    loss_avg = AverageMeter()

    for epoch in range(args.num_train_epochs):
        train_one_epoch(
            unet,
            text_encoder,
            accelerator,
            train_dataloader,
            noise_scheduler,
            optimizer_unet,
            optimizer_text_encoder,
            lr_scheduler_text_encoder,
            lr_scheduler_unet,
            loss_avg,
            progress_bar,
            args,
            ticker,
        )

        accelerator.wait_for_everyone()
