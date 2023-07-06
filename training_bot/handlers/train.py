import asyncio
import os
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

from aiogram import Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder

from logs import get_train_bot_logger
from logs import log_function
from trainer import create_model
from trainer.utils.Ticker import Ticker
from training_bot.utils.Queue import PriorityQueue
from training_bot.utils.states import Train

router = Router()

logger = get_train_bot_logger()
queue = PriorityQueue()
images_path = "./images/"
result_path = "./result/"

TOTAL_STEPS = 3000


@log_function(logger, 'Training model')
def train_task(user_id: int, username: str, user_class: str, conn: Connection):
    user_folder = os.path.join(images_path, f"{user_id}")
    result_folder = os.path.join(result_path, f"{username}")
    logger.info(f"Start training for user {username}")
    
    class BotTicker(Ticker):
        def tick(self, data: str = ''):
            conn.send(data)
            return super().tick()

    try:
        create_model(
            instance_data_dir=user_folder,
            instance_class=user_class,
            repo_name=f"{username}",
            output_dir=result_folder,
            max_train_steps=TOTAL_STEPS,
            ticker=BotTicker()
        )
    except Exception as e:
        logger.error(str(e))
        raise e


@router.message(Command(commands=["train"]), StateFilter(None))
async def hello_message(message: Message, state: FSMContext):
    user_id = message.from_user.id
    username = message.from_user.username

    data = message.text.split(maxsplit=1)[1:]
    if data:
        username = data[0].replace(" ", "_")

    user_folder = os.path.join(images_path, f"{user_id}")
    if not os.path.exists(user_folder) or not os.listdir(user_folder):
        await message.answer("А датасет где?")
        return

    queue.add(user_id)
    await state.set_state(Train.training)
    logger.info(f"Added {username} to training queue")

    keyboard = InlineKeyboardBuilder()
    keyboard.button(text="отмена", callback_data="cancel")

    prev_position = queue.find(user_id)
    msg = await message.answer(
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
            await msg.edit_text(
                f"Добавил в очередь на обучение\nПозиция в очереди:\t{prev_position + 1}"
            )
        await asyncio.sleep(0.1)

    await msg.edit_text("Подготовка...", reply_markup=None)
    # create train process
    
    conn, child = Pipe()

    process = Process(target=train_task, args=(user_id, username, "man", child))
    process.start()

    await msg.edit_text("Обучение...", reply_markup=None)

    while process.is_alive():
        await asyncio.sleep(0.2)
        if conn.poll():
            step = conn.recv()
            if step != TOTAL_STEPS:
                await msg.edit_text(f"Обучение...\nШаг {step}/{TOTAL_STEPS}")
            else:
                await msg.edit_text(f"Загрузка на сервер")

    # drop user_id from queue
    queue.pop()
    await state.clear()

    if process.exitcode:
        await msg.edit_text("Произошла ошибка, мы уже решаем")
    else:
        await msg.edit_text("Обучение завершено, модель загружена на сервер.")


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
