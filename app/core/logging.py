import logging
import logging.handlers
import queue
from pathlib import Path
from app.core.config import settings


_listener = None


def setup_logging() -> None:
    global _listener
    if _listener is not None:
        return

    log_dir: Path = settings.log_dir_abs
    log_file = log_dir / "app.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=settings.log_max_bytes, backupCount=settings.log_backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    log_queue: queue.Queue = queue.Queue(maxsize=10000)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root.addHandler(queue_handler)

    _listener = logging.handlers.QueueListener(log_queue, stream_handler, file_handler, respect_handler_level=True)
    _listener.start()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
