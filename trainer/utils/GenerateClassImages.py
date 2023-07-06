import hashlib
from pathlib import Path

import torch
import torch.utils.checkpoint
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from tqdm.auto import tqdm

from logs import get_train_logger
from logs import log_function
from trainer.datasets import PromptDataset

logger = get_train_logger()


@log_function(logger, 'Generating class images')
def generate_class_images(args, accelerator):
    pipeline = None
    for concept in args.concepts_list:
        class_images_dir = Path(concept["class_data_dir"])
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if pipeline is None:
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=args.revision
                )
                pipeline.set_progress_bar_config(disable=True)
                pipeline.to(accelerator.device)
                pipeline.enable_xformers_memory_efficient_attention()
                pipeline.unet.to(memory_format=torch.channels_last)
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

            num_new_images = args.num_class_images - cur_class_images

            sample_dataset = PromptDataset(concept["class_prompt"], num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)

            with torch.autocast("cuda"), torch.inference_mode():
                for example in tqdm(
                        sample_dataloader, desc="Generating class images",
                        disable=not accelerator.is_local_main_process
                ):
                    images = pipeline(example["prompt"], num_inference_steps=40).images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

    del pipeline

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
