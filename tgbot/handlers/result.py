import io
import json
import os

import numpy as np
from aiogram import Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, InputMediaPhoto, BufferedInputFile

from logs import get_train_bot_logger
from tgbot.utils.gen_image import generate_images

logger = get_train_bot_logger()

router = Router()

result_path = "./result/"

negative_man = "feminine, swollen, blurry, out of focus, slanting eyes, black and white, deformed, asymmetrical face, " \
               "bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, missing limb, " \
               "floating limbs, disconnected limbs, malformed hands, long neck, long body"

negative_woman = "masculine, swollen, blurry, out of focus, slanting eyes, black and white, deformed, asymmetrical " \
                 "face, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, missing limb, " \
                 "floating limbs, disconnected limbs, malformed hands, long neck, long body"


@router.message(Command(commands=["generate"]), StateFilter(None))
async def generate(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_folder = os.path.join(result_path, f"{message.from_user.id}", "3000")

    if not os.path.exists(user_folder):
        await message.answer("Вашей модели не существует в базе")
        return

    logger.info(f"Exported {user_folder}")

    await message.answer("Генерация...")

    cls = "xyz man"
    negative = negative_man

    with open(f"./result/{user_id}/3000/args.json", 'r') as f:
        path = json.loads(f.read())

    gender_info = path["instance_class"]
    if gender_info == "woman":
        cls = "xyz woman"
        negative = negative_woman

    with open(f"./tgbot/prompts/prompt.json", 'r') as f:
        prompts = json.loads(f.read())['themes']

    negative_prompts = [negative for _ in range(len(prompts))]
    seeds = list(map(int, np.random.randint(1, 2147483647, size=len(prompts))))
    n = 3

    for batch in generate_images(cls, prompts, negative_prompts, user_folder, seeds, n):
        contents = []

        for image in batch:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            contents.append(
                InputMediaPhoto(
                    media=BufferedInputFile(img_byte_arr.getvalue(), filename='image.png')
                )
            )

            if len(contents) == 10:
                await message.answer_media_group(contents)
                contents = []

        await message.answer_media_group(contents)

    await message.answer("Как вам результаты?")
