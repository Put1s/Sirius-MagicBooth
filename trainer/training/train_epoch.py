import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from trainer.utils.SaveWeights import save_weights
from trainer.utils.Ticker import Ticker


def train_one_epoch(
        unet,
        text_encoder,
        accelerator,
        dataloader,
        noise_scheduler,
        optimizer_unet,
        optimizer_text_encoder,
        lr_scheduler_text_encoder,
        lr_scheduler_unet,
        loss_avg,
        progress_bar,
        args,
        ticker: Ticker,
):
    unet.train()

    # True if text encoder needs training
    text_encoder_training = args.global_step < args.max_train_text_encoder

    if text_encoder_training:
        text_encoder.train()

    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(unet):
            # Convert images to latent space
            latent_dist = batch[0][0]
            latents = latent_dist.sample() * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=accelerator.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch[0][1])[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = (
                F.mse_loss(model_pred.float(), target.float(), reduction="none")
                .mean([1, 2, 3])
                .mean()
            )

            # Compute prior loss
            prior_loss = F.mse_loss(
                model_pred_prior.float(), target_prior.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = loss + args.prior_loss_weight * prior_loss

            accelerator.backward(loss)

            optimizer_unet.step()
            lr_scheduler_unet.step()
            optimizer_unet.zero_grad(set_to_none=True)

            if text_encoder_training:
                optimizer_text_encoder.step()
                lr_scheduler_text_encoder.step()
                optimizer_text_encoder.zero_grad(set_to_none=True)

            loss_avg.update(loss.detach_(), bsz)

        if not args.global_step % args.log_interval:
            logs = {
                "loss": loss_avg.avg.item(),
                "lr": lr_scheduler_unet.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
        if (
                args.global_step > 0
                and not args.global_step % args.save_interval
                and args.global_step >= args.save_min_steps
        ):
            save_weights(accelerator, unet, text_encoder, args)

        progress_bar.update(1)

        args.global_step += 1
        ticker.tick(args.global_step)

        # switch text_encoder in eval mode
        if args.global_step == args.max_train_text_encoder:
            text_encoder.requires_grad_(False)
            text_encoder.eval()

            del optimizer_text_encoder
            del lr_scheduler_text_encoder

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            text_encoder_training = False

        if args.global_step >= args.max_train_steps:
            break
