import json

import torch
from accelerate.utils import set_seed

from trainer.training import train, init_models
from trainer.utils.ParseArgs import parse_args
from trainer.utils.SaveWeights import save_weights
from trainer.utils.Ticker import StepCounter

torch.backends.cudnn.benchmark = True


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

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

    (
        accelerator,
        unet,
        text_encoder,
        optimizer_unet,
        optimizer_text_encoder,
        lr_scheduler_unet,
        lr_scheduler_text_encoder,
        train_dataloader,
        noise_scheduler,
    ) = init_models(args)

    args.global_steps = StepCounter()
    # !!!Train!!!
    train(
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
    )

    # Save final model
    save_weights(accelerator, unet, text_encoder, args)

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
