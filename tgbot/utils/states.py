from aiogram.fsm.state import StatesGroup, State


class Train(StatesGroup):
    choosing_class = State()
    training = State()
