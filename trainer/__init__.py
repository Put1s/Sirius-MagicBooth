from pathlib import Path

from logs import get_train_logger
from logs import log_function
from trainer.training import init_models, train
from trainer.utils.Dict2Object import ObjectGenerator
from trainer.utils.SaveWeights import save_weights
from trainer.utils.Ticker import Ticker

logger = get_train_logger()


# TODO: return model instead of path
@log_function(logger, 'Creating personal model')
def create_model(
        instance_data_dir: str,
        instance_name: str = "xyz",
        instance_class: str = "man",
        class_images: int = 400,
        pretrained_model: str = "stabilityai/stable-diffusion-2-1-base",
        output_dir: str = "results",
        mixed_precision: str = 'fp16',
        prior_loss_weight: float = 1.0,
        seed: int = 1234,
        resolution: int = 512,
        learning_rate: float = 1e-6,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 500,
        max_train_steps: int = 3000,
        max_train_text_encoder: int = 1500,
        save_interval: int = 10000,
        save_min_steps: int = 5000,
        revision: str = 'fp16',
        repo_name: str = "mans",
        ticker: Ticker=Ticker()
) -> str:
    """Create model model from instance's images.

    Args:
        instance_data_dir (Path): Images of instance.
        instance_name (str, optional): Keyword for instance. Defaults to 'xyz'.
        instance_class (str, optional): Instance's class. Defaults to "man".
        class_images (int, optional): Amount of class images to train. Defaults to 400.
        pretrained_model (str, optional): Name or path to pretrained model. Defaults to "stabilityai/stable-diffusion-2-1-base".
        output_dir (Path, optional): Dirname for output weights. Defaults to "results".
        mixed_precision (str, optional): _description_. Defaults to 'fp16'.
        prior_loss_weight (float, optional): _description_. Defaults to 1.0.
        seed (int, optional): Random seed for training. Defaults to 1234.
        resolution (int, optional): Final image resolution. Defaults to 512.
        learning_rate (float, optional): Defaults to 1e-6.
        lr_scheduler (str, optional): _description_. Defaults to "constant".
        lr_warmup_steps (int, optional): Number of warming up steps. Defaults to 500.
        max_train_steps (int, optional): Total amount of training steps. Defaults to 3000.
        max_train_text_encoder (int, optional): Amount of training steps for text encoder. Defaults to 1500.
        save_interval (int, optional): Interval in steps beetween saving intermediate weights. Defaults to 500.
        save_min_steps (int, optional): _description_. Defaults to 500.
        revision (str, optional): Defaults to None.
        repo_name (str, optional): Name of repository to save the model. No Defaults to None.
    Returns:
        str: path to the weights of trained model
    """

    args = ObjectGenerator(
        pretrained_model_name_or_path=pretrained_model,
        revision=revision,
        instance_data_dir=f'{instance_data_dir}',
        class_data_dir=f'images/{instance_class}',
        instance_prompt=f'{instance_name} {instance_class}',
        class_prompt=f'a {instance_class}',
        save_sample_prompt=None,
        save_sample_negative_prompt=None,
        n_save_sample=5,
        save_guidance_scale=7.5,
        save_infer_steps=50,
        pad_tokens=False,
        prior_loss_weight=prior_loss_weight,
        num_class_images=class_images,
        output_dir=output_dir,
        seed=seed,
        resolution=resolution,
        center_crop=False,
        train_batch_size=1,
        sample_batch_size=5,
        max_train_steps=max_train_steps,
        max_train_text_encoder=max_train_text_encoder,
        gradient_checkpointing=False,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=1e-2,
        adam_epsilon=1e-08,
        max_grad_norm=1.0,
        log_interval=10,
        save_interval=save_interval,
        save_min_steps=save_min_steps,
        mixed_precision=mixed_precision,
        concepts_list=None,
        repo_name=repo_name,
        global_step=0,
        instance_class=instance_class,
    )

    # Initialize models
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

    # Train the model
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
        ticker,
    )

    # Save model's weights
    weights_path = save_weights(accelerator, unet, text_encoder, args, load_repo=True)

    return weights_path


if __name__ == "__main__":
    # create_model(
    #     instance_data_dir='images/299355675',
    #     instance_name="xyz",
    #     instance_class="man",
    #     pretrained_model="results/gachi"
    # )

    create_model(
        instance_data_dir='images/241301944',
        instance_name="xyz",
        instance_class="man"
    )
