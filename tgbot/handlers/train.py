import asyncio
import os
from multiprocessing import Process

from aiogram import Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder, InlineKeyboardButton

from logs import get_train_bot_logger
from logs import log_function
from tgbot.utils.Queue import PriorityQueue
from tgbot.utils.states import Train
from trainer import create_model

router = Router()

logger = get_train_bot_logger()
queue = PriorityQueue()
images_path = "./images/"
result_path = "./result/"

PUBLIC_RUN = True


@log_function(logger, 'Training model')
def train_task(user_id: int, cls: str):
    user_folder = os.path.join(images_path, f"{user_id}")
    result_folder = os.path.join(result_path, f"{user_id}")
    logger.info(f"Start training for user {user_id} : {cls}")

    try:
        create_model(
            instance_class=cls,
            instance_data_dir=user_folder,
            repo_name=f"{user_id}",
            output_dir=result_folder,
        )
    except Exception as e:
        logger.error(str(e))
        raise e


@router.message(Command(commands=["train"]), StateFilter(None))
async def train(message: Message, state: FSMContext):
    username = message.from_user.username
    user_id = message.from_user.id
    data = message.text.split(maxsplit=1)[1:]
    if data:
        username = data[0].replace(" ", "_")
    user_folder = os.path.join(images_path, f"{user_id}")
    if not os.path.exists(user_folder) or len(os.listdir(user_folder)) < 15:
        await message.answer("Фотографий слишком мало!")
        return
    await state.update_data(username=username)
    keyboard = InlineKeyboardBuilder()
    keyboard.add(InlineKeyboardButton(text="Мужчина", callback_data="man"),
                 InlineKeyboardButton(text="Женщина", callback_data="woman"))
    await message.answer("Выберите класс", reply_markup=keyboard.as_markup())
    await state.set_state(Train.choosing_class)


@router.callback_query(StateFilter(Train.choosing_class))
async def callback_query(query: CallbackQuery, state: FSMContext):
    user_id = query.from_user.id
    username = (await state.get_data())["username"]
    message = query.message
    cls = query.data

    queue.add(user_id)
    await state.set_state(Train.training)
    logger.info(f"Added {username} to training queue")

    keyboard = InlineKeyboardBuilder()
    keyboard.button(text="отмена", callback_data="cancel")

    prev_position = queue.find(user_id)

    await message.edit_text(
        f"Добавил в очередь на обучение\nПозиция в очереди:\t{prev_position}",
        reply_markup=keyboard.as_markup(),
    )

    # wait until user_id is on the top of the queue
    while queue.get() != user_id:
        position = queue.find(user_id)
        if position is None:  # if cancel button was pressed and user_id not in queue
            return

        if position != prev_position:  # if position in queue has changed
            prev_position = position
            await message.edit_text(
                f"Добавил в очередь на обучение\nПозиция в очереди:\t{prev_position + 1}"
            )
        await asyncio.sleep(0.1)

    await message.edit_text("Подготовка...")
    # create train process

    if PUBLIC_RUN:
        await message.edit_text("Из-за больших нагрузок на сервер в данный момент мы отключили обучение моделей.\nСпасибо за понимание.\n")
        # await message.edit_text("Обучение завершено, модель загружена на сервер.")
        await state.clear()
        await asyncio.sleep(5)
        return
    
    process = Process(target=train_task, args=(user_id, cls))
    process.start()

    await message.edit_text('Обучение идет...')

    while process.is_alive():
        await asyncio.sleep(0.2)

    # drop user_id from queue
    queue.pop()
    await state.clear()

    if process.exitcode:
        await message.answer("Произошла ошибка, мы уже решаем")
    else:
        await message.answer("Обучение завершено!")

    await message.delete()


@router.callback_query(StateFilter(Train.training))
async def hello_message(query: CallbackQuery, state: FSMContext):
    user_id = query.from_user.id

    # check if user_id
    if queue.find(user_id):
        queue.remove(user_id)
        logger.info(f"Deleted {user_id} from train queue.")

        await state.clear()
        await query.message.delete()

    elif queue.get() == user_id:
        await query.answer("Обучение уже началось.")
