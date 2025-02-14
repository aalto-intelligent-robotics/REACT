import os
import datetime
import logging


class LogFormatter(logging.Formatter):
    """
    Logging colored formatter, adapted from
    https://stackoverflow.com/a/56944256/3638629
    """

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + "%(message)s" + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset + "%(message)s",
            logging.WARNING: self.yellow + self.fmt + "%(message)s" + self.reset,
            logging.ERROR: self.red + self.fmt + "%(message)s" + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + "%(message)s" + self.reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def getLogger(
    name: str,
    consoleLevel: int = logging.INFO,
    fileLevel: int = logging.DEBUG,
    fmt: str = "%(asctime)s - %(name)s - [%(levelname)s]: ",
    log_path: str = f"/home/ros/logs/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
    log_file: str = f"default.log",
) -> logging.Logger:
    # Set up logger
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # stdout handler for logging to the console
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(consoleLevel)
    stdout_handler.setFormatter(LogFormatter(fmt))
    # File handler for logging to a file
    file_handler = logging.FileHandler(os.path.join(log_path, log_file))
    file_handler.setLevel(fileLevel)
    file_handler.setFormatter(logging.Formatter(fmt + "%(message)s"))
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger
