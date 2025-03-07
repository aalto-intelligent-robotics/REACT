import os
import datetime
import logging


class LogFormatter(logging.Formatter):
    """
    Logging colored formatter, adapted from
    https://stackoverflow.com/a/56944256/3638629.

    This class provides a logging formatter that adds colors to log messages
    based on their severity level.
    """

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        """Initialize the LogFormatter with the given format.

        :param fmt: The format string to be used for log messages.
        """
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
        """Format the log record with the appropriate color based on its
        severity level.

        :param record: The log record to be formatted.
        :return: The formatted log message.
        :rtype: str
        """
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
    """Set up and return a logger with both console and file handlers.

    This function sets up a logger with specified name, console logging level, file
    logging level, format, log path, and log file name.
    It configures the logger to log messages both to the console (with colored
    formatting) and to a file.

    :param name: The name of the logger.
    :param consoleLevel: The logging level for the console handler.
        Default is logging.INFO.
    :param fileLevel: The logging level for the file handler.
        Default is logging.DEBUG.
    :param fmt: The format string to be used for log messages.
        Default is "%(asctime)s - %(name)s - [%(levelname)s]: ".
    :param log_path: The path where log files will be saved.
        Default is a directory named with the current timestamp.
    :param log_file: The name of the log file.
        Default is "default.log".
    :return: The configured logger.
    """
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
