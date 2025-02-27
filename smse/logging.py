import datetime as dt
import json
import logging
import logging.config
import os

from dotenv import load_dotenv

load_dotenv()

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


class MultiLineFormatter(logging.Formatter):
    """Custom formatter that properly pads multi-line messages."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)

        # For multi-line messages, add padding to each line after the first
        if "\n" in message:
            # Split the message into lines
            lines = message.split("\n")

            # Get the prefix from the first line (everything before the actual message)
            first_line = lines[0]
            message_start = first_line.find(" - ")
            if message_start >= 0:
                prefix = " " * (message_start + 4)  # +3 for ' - '

                # Add the prefix to each subsequent line
                lines = [lines[0]] + [prefix + line for line in lines[1:]]

                # Join the lines back together
                message = "\n".join(lines)

        # Align the filename at the end of the line
        last_open_bracket = message.rfind("(")
        if last_open_bracket != -1:
            padding_length = (
                80 - last_open_bracket
            )  # Adjust 80 to your desired line length
            if padding_length > 0:
                message = (
                    message[:last_open_bracket]
                    + " " * padding_length
                    + message[last_open_bracket:]
                )

        return message


class JSONFormatter(logging.Formatter):
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
            "()": "smse.logging.MultiLineFormatter",
            "format": "[%(asctime)s] [%(levelname)8s] -  %(message)s (%(filename)s:%(lineno)s)",  # noqa: E501
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
        "json": {
            "()": "smse.logging.JSONFormatter",
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
    },
    "loggers": {"root": {"level": "DEBUG", "handlers": ["stdout", "file_json"]}},
}


def setup_logging() -> None:
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
