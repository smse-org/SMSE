import datetime as dt
import json
import logging
import logging.config
import os

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}

LOG_DIR = os.getenv("LOG_DIR", ".logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")


class MyJSONFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict[str, str]:
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: (
                msg_val
                if (msg_val := always_fields.pop(val, None)) is not None
                else getattr(record, val)
            )
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


# Logger configuration as a Python dictionary
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] [%(levelname)8s] -  %(message)s (%(name)s:%(filename)s:%(lineno)s)",  # noqa: E501
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
        "json": {
            "()": "smse.logging.MyJSONFormatter",
            "fmt_keys": {
                "level": "levelname",
                "message": "message",
                "timestamp": "timestamp",
                "logger": "name",
                "module": "module",
                "function": "funcName",
                "line": "lineno",
                "thread_name": "threadName",
            },
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file_json": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": f"{LOG_DIR}/smse.log.jsonl",
            "maxBytes": 1024 * 1024 * 10,  # 10 MB
            "backupCount": 3,
        },
        # "queue_handler": {
        #     "class": "logging.handlers.QueueHandler",
        #     "handlers": ["stdout", "file_json"],
        #     "respect_handler_level": True,
        # },
    },
    "loggers": {"root": {"level": "DEBUG", "handlers": ["stdout", "file_json"]}},
}


def setup_logging() -> None:
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logging.config.dictConfig(LOGGING_CONFIG)

    # queue_handler = logging.getHandlerByName("queue_handler")
    # if queue_handler is not None:
    #     queue_handler.listener.start()
    #     atexit.register(queue_handler.listener.stop)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
