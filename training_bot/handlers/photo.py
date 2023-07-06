import os
from random import choice

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

    image = message.photo[-1]
    image_file = await bot.get_file(image.file_id)
    file = await bot.download_file(image_file.file_path)

    img = Image.open(file)
    width, height = img.size

    if width != height:
        await message.answer('Я принимаю только квадратные изображения')
        return
    if width < 100:
        await message.answer('Слишком маленькое разрешение')
        return

    images_count = len(os.listdir(user_folder))
    image_path = os.path.join(user_folder, f'{images_count + 1:>04}.png')

    img.save(image_path)

    logger.info(f'Downloaded image: {image_path}')
    stickers = [
        'CAACAgIAAxkBAAEG3vJjnj9M6bW_Nhxm7j8i4F3lkUJWbQACMgADCSrWG5gG65wywIWcLAQ',
        'CAACAgIAAxkBAAEG3vBjnj9GU8T157L8i_As0M6Rc9_cdwACwBcAAkqxaEniHojmZajCXiwE',
        'CAACAgIAAxkBAAEG3u5jnj881aB9jbe_MFIj6CiSNouf3AAC_xUAArPK-Eh5o60GJl1xCywE',
        'CAACAgIAAxkBAAEG3uxjnj85MEtIFl1QzcEb-auHqJsy9QACbRQAAvh48Ev_35tLbqKxRywE',
        'CAACAgIAAxkBAAEG3vZjnj_oXCZkvglU37rdNsc0EwPm0QACDA8AAkNFQEizvdi9hSfKWSwE',
        'CAACAgIAAxkBAAEG3wJjnkI87lT_z4cxd8xqk3eC9k4CIQACUhMAAjFuOUgEv3MHYx5zkywE'
    ]
    await message.reply_sticker(sticker=choice(stickers))
