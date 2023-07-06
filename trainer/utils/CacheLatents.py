import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

from logs import get_train_logger
from logs import log_function
from trainer.datasets import LatentsDataset, DreamBoothDataset

logger = get_train_logger()


@log_function(logger, 'Caching latents')
def cache_latents(args, accelerator):
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_auth_token=True
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        use_auth_token=True
    ).requires_grad_(False).to(torch.float16)

    train_dataset = DreamBoothDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        pad_tokens=args.pad_tokens,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        ).input_ids

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    vae.to(accelerator.device)

    latents_cache = []
    text_encoder_cache = []

    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=torch.float16)
            batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)

            latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

            text_encoder_cache.append(batch["input_ids"])

    train_dataset = LatentsDataset(latents_cache, text_encoder_cache)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=lambda x: x,
        shuffle=True
    )

    del vae
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_dataloader
