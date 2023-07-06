import json
import os

import torch
import torch.utils.checkpoint
from diffusers import DiffusionPipeline
from huggingface_hub import HfApi, create_repo

from logs import get_train_logger
from logs import log_block
from logs import log_function

logger = get_train_logger()


@log_function(logger, 'Saving weights')
def save_weights(accelerator, unet, text_encoder, args, load_repo: bool = False) -> str:
    """
    Save the weights of the unet and returns path to them.

    :param accelerator: The accelerator to use.
    :param unet: The unet.
    :param text_encoder: The text encoder.
    :param args: The arguments."""

    # Create the pipeline using the trained modules and save it.
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        text_encoder=accelerator.unwrap_model(text_encoder),
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(torch.float16)

    # folder path
    save_dir = os.path.join(args.output_dir, f"{args.global_step}")

    # save model and args
    pipeline.save_pretrained(save_dir)

    args_dict = args.__dict__
    args_dict.pop('global_step', None)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
        logger.info(f"Weights saved at {save_dir}")

    if load_repo and args.repo_name:
        with log_block(logger, 'Creating cloud repository'):
            create_repo(
                f"DreamBoothSirius/{args.repo_name}",
                repo_type="model",
                private=True,
                exist_ok=True,
            )
        with log_block(logger, 'Uploading model to cloud repository'):
            api = HfApi()
            api.upload_folder(
                folder_path=save_dir,
                path_in_repo=".",
                repo_id=f"DreamBoothSirius/{args.repo_name}",
                repo_type="model",
            )
    elif load_repo:
        logger.warning(
            "Unable to load model to repository. No repository name specified."
        )

    return save_dir
