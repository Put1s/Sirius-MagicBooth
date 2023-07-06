import os

from PIL import Image
from aiogram import Router, F, Bot
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from logs import get_train_bot_logger

logger = get_train_bot_logger()

router = Router()

images_path = './images/'


@router.message(F.photo, StateFilter(None))
async def msg(message: Message, bot: Bot, state: FSMContext):
    user_id = message.from_user.id

    user_folder = os.path.join(images_path, f'{user_id}')
    os.makedirs(user_folder, exist_ok=True)
    images_count = len(os.listdir(user_folder))
    
    if images_count >= 50:
        await message.answer("Вы загрузили слишком много фотографий")
        return
        

    image = message.photo[-1]
    image_file = await bot.get_file(image.file_id)
    file = await bot.download_file(image_file.file_path)

    img = Image.open(file)
    width, height = img.size

    if width != height:
        await message.answer('Допустимы только квадратные изображения')
        return
    if width < 100:
        await message.answer('Разрешение слишком маленькое')
        return

    image_path = os.path.join(user_folder, f'{images_count + 1:>04}.png')

    img.save(image_path)

    logger.info(f'Downloaded image: {image_path}')
    await message.reply("Фотография загружена")
