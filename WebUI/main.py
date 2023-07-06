import gradio as gr
import torch
from diffusers import DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline
from torch import autocast

negative = "feminine, swollen, blurry, out of focus, slanting eyes, deformed, asymmetrical face, bad anatomy, " \
           "disfigured, poorly drawn face, mutation, extra limb, ugly, missing limb, long neck, long body "

model_path = "DreamBoothSirius/d0gied"
model_paths = {
    "Bogdan": "DreamBoothSirius/put1s",
    "German": "DreamBoothSirius/d0gied",
    "Marat": "DreamBoothSirius/Capi2list",
    "Nikolay": "DreamBoothSirius/khadzakos"
}
model_paths_inpaint = {
    "Stable Diffusion In-painting": "stabilityai/stable-diffusion-2-inpainting"
}

height = 512
width = 512

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

pipelines = {
    model_paths[key]: DiffusionPipeline.from_pretrained(
        model_paths[key],
        torch_dtype=torch.float16
    ) for key in model_paths.keys()
}

for _ in pipelines.keys():
    pipelines[_].enable_xformers_memory_efficient_attention()
    pipelines[_].unet.to(memory_format=torch.channels_last)
    pipelines[_].scheduler = DPMSolverMultistepScheduler.from_config(pipelines[_].scheduler.config)

pipelines_inpaint = {
    model_paths_inpaint[key]: DiffusionPipeline.from_pretrained(
        model_paths_inpaint[key],
        revision='fp16',
        torch_dtype=torch.float16
    ) for key in model_paths_inpaint.keys()
}

for _ in pipelines_inpaint.keys():
    pipelines_inpaint[_].enable_xformers_memory_efficient_attention()
    pipelines_inpaint[_].unet.to(memory_format=torch.channels_last)
    pipelines_inpaint[_].scheduler = DPMSolverMultistepScheduler.from_config(pipelines_inpaint[_].scheduler.config)

pipeline = pipelines[model_path].to('cuda')


@autocast('cuda')
@torch.inference_mode()
def generate(prompt, negative_prompt, model, num_of_images, random, gscale):
    global model_path
    global pipeline

    valid_negative_prompt = negative
    if negative_prompt is not None:
        valid_negative_prompt += ', ' + negative_prompt

    if model_paths[model] != model_path:
        model_path = model_paths[model]
        del pipeline
        torch.cuda.empty_cache()
        pipeline = pipelines[model_path].to('cuda')

    g_cuda = torch.Generator(device="cuda")
    seed = int(random if random else '1234')
    g_cuda.manual_seed(seed)

    images = pipeline(
        prompt,
        height=height,
        width=width,
        negative_prompt=valid_negative_prompt,
        num_images_per_prompt=int(num_of_images),
        num_inference_steps=30,
        guidance_scale=gscale,
        generator=g_cuda,
    ).images

    return images


@autocast('cuda')
@torch.inference_mode()
def generate_inpaint(prompt, negative_prompt, img_dict, model, num_of_images, random, gscale):
    global model_path
    global pipeline

    valid_negative_prompt = negative
    if negative_prompt is not None:
        valid_negative_prompt += ', ' + negative_prompt

    if model_paths_inpaint[model] != model_path:
        model_path = model_paths_inpaint[model]
        del pipeline
        torch.cuda.empty_cache()
        pipeline = pipelines_inpaint[model_path].to('cuda')

    init_img = img_dict['image'].convert("RGB").resize((512, 512))
    mask_img = img_dict['mask'].resize((512, 512))

    g_cuda = torch.Generator(device="cuda")
    seed = int(random if random else '1234')
    g_cuda.manual_seed(seed)

    images = pipeline(
        prompt,
        image=init_img,
        mask_image=mask_img,
        negative_prompt=valid_negative_prompt,
        num_images_per_prompt=int(num_of_images),
        num_inference_steps=30,
        guidance_scale=gscale,
        generator=g_cuda,
    ).images

    return images


with gr.Blocks(title="MagicBooth") as booth:
    with gr.Tab("Classic txt2img"):
        with gr.Group():
            with gr.Row().style(equal_height=True):
                with gr.Column():
                    prompts1 = [
                        gr.Textbox(
                            label="Prompt",
                            placeholder="Input your prompt",
                            max_lines=1
                        ),
                        gr.Textbox(
                            label="Negative prompt",
                            placeholder="Input your negative prompt or leave it empty",
                            max_lines=1
                        )
                    ]
                btn1 = gr.Button("Generate").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False
                )
        with gr.Accordion("Advanced options", open=False):
            adv1 = [
                gr.Radio(
                    list(model_paths.keys()),
                    label="Models",
                    value="German"
                ),
                gr.Radio(
                    list(map(str, range(1, 6))),
                    value='1',
                    label="Number of samples"
                ),
                gr.Slider(
                    1,
                    2147483647,
                    step=1,
                    value=1,
                    label="Random seed",
                    randomize=True
                ),
                gr.Slider(
                    1,
                    10,
                    value=7.5,
                    step=0.5,
                    label="Guidance scale"
                )
            ]
        out1 = gr.Gallery(show_label=False)

        gr.Examples(
            [
                [
                    "cowboy xyz man, portrait painting, looking in the camera, best quality, cowboy leather hat",
                    ""
                    ],
                 [
                     "handsome rdr2 xyz man, awesome from the chest up portrait painting, 4k, hyperrealism, posing",
                     ""
                 ],
                 [
                     "ancient greek god xyz man, awesome from the chest up portrait painting, 4k, hyperrealism, "
                     "posing, curly hair, enlightened",
                     ""
                 ],
                 [
                     "astronaut xyz man, awesome from the chest up portrait painting, 4k, hyperrealism, posing, "
                     "face uncovered",
                     "helmet"
                 ],
                 [
                     "handsome xyz man, awesome from the chest up portrait painting, oil, 55mm, 1950, best quality,  "
                     "4k, hyperrealism, posing, black and white, vignette",
                     ""
                 ],
                [
                    "handsome xyz man sith holding a laser sword, awesome from the chest up portrait painting, "
                    "Star Wars, best quality,  4k, hyperrealism, posing",
                    "darth vader"
                ]
            ],
            prompts1,
        )

        btn1.click(fn=generate, inputs=[*prompts1, *adv1], outputs=out1)
    with gr.Tab("Inpaint img2img"):
        with gr.Group():
            image = gr.Image(source='upload', tool='sketch', type='pil').style(width=512, height=512)
            with gr.Row().style(equal_height=True):
                with gr.Column():
                    prompts2 = [
                        gr.Textbox(
                            label="Prompt",
                            placeholder="Input your prompt",
                            max_lines=1
                        ),
                        gr.Textbox(
                            label="Negative prompt",
                            placeholder="Input your negative prompt or leave it empty",
                            max_lines=1
                        )
                    ]
                btn2 = gr.Button("Generate").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False
                )
        with gr.Accordion("Advanced options", open=False):
            adv2 = [
                gr.Radio(
                    ["Stable Diffusion In-painting"],
                    label="Models",
                    value="Stable Diffusion In-painting",
                    visible=False,
                ),
                gr.Radio(
                    list(map(str, range(1, 6))),
                    value='1',
                    label="Number of samples"
                ),
                gr.Slider(
                    1,
                    2147483647,
                    step=1,
                    value=1,
                    label="Random seed",
                    randomize=True
                ),
                gr.Slider(
                    1,
                    10,
                    value=7.5,
                    step=0.5,
                    label="Guidance scale"
                )
            ]

        out2 = gr.Gallery(show_label=False)

        btn2.click(fn=generate_inpaint, inputs=[*prompts2, image, *adv2], outputs=out2)

booth.queue()
booth.launch(share=True)
