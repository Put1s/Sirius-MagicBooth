import functools
from contextlib import contextmanager

from loguru import logger


def log_function(current_logger, message: str = None, level: str = "INFO"):
    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            current_logger.log(level, "[START] '{}' - [{}]", name, message)

            result = func(*args, **kwargs)

            current_logger.log(level, "[END] '{}' - [{}]", name, message)

            return result

        return wrapped

    return wrapper


@contextmanager
def log_block(current_logger, message: str = None, level: str = "INFO"):
    current_logger.log(level, "[START] - [{}]", message)
    yield
    current_logger.log(level, "[END] - [{}]", message)


class Logger:
    train_logger: logger
    train_bot_logger: logger
    is_inited = False

    @classmethod
    def init(cls):
        if not cls.is_inited:
            cls.train_logger = logger.bind(task="training")
            cls.train_bot_logger = logger.bind(task="train_bot")
            logger.add("train_logs.log", enqueue=True, filter=lambda x: x["extra"]["task"] == "training")
            logger.add("train_bot_logs.log", enqueue=True, filter=lambda x: x["extra"]["task"] == "train_bot")
        cls.is_inited = True

    @classmethod
    def get_train_logger(cls):
        if not cls.is_inited:
            cls.init()
        return cls.train_logger

    @classmethod
    def get_train_bot_logger(cls):
        if not cls.is_inited:
            cls.init()
        return cls.train_bot_logger


def get_train_logger():
    return Logger.get_train_logger()


def get_train_bot_logger():
    return Logger.get_train_bot_logger()


if __name__ == "__main__":
    logger.debug("сообщение для отладки")
    logger.info("информационное сообщение")
    logger.warning("предупреждающее сообщение")
    logger.error("сообщение об ошибке")
    logger.critical("сообщение о критической ошибке")
