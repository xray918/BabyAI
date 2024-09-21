import logging
import sys
import os


class ColoredFormatter(logging.Formatter):
    """自定义的日志格式化器，添加颜色支持"""

    # ANSI 转义码
    COLOR_MAP = {
        'DEBUG': '\033[96m',     # 青色
        'INFO': '\033[92m',      # 绿色
        'WARNING': '\033[93m',   # 黄色
        'ERROR': '\033[95m',     # 洋红色
        'CRITICAL': '\033[91m'   # 红色
    }
    RESET_COLOR = '\033[0m'

    def format(self, record):
        log_color = self.COLOR_MAP.get(record.levelname, self.RESET_COLOR)
        record.msg = f"{log_color}{record.msg}{self.RESET_COLOR}"
        return super().format(record)

def _reset_logger(log):
    for handler in log.handlers:
        handler.close()
        log.removeHandler(handler)
    log.handlers.clear()
    log.propagate = False

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        "[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    log.addHandler(console_handler)

    # 文件处理器
    if getattr(sys, 'frozen', False):
        app_path = os.path.dirname(sys.executable)
    else:
        app_path = os.path.dirname(os.path.abspath(__file__)) + "/.."

    log_file = f"{app_path}/account/run.log"
    path = os.path.dirname(log_file)
    if not os.path.exists(path):
        os.makedirs(path)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_formatter = logging.Formatter(
        "[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    log.addHandler(file_handler)

def _get_logger():
    log = logging.getLogger("log")
    _reset_logger(log)
    log.setLevel(logging.DEBUG)
    return log

# 日志句柄
logger = _get_logger()