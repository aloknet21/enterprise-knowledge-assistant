import os


DEFAULT_THREADS = min(2, os.cpu_count() or 1)
DEFAULT_BATCH_SIZE = 8


def validate_threads(num_threads: int) -> None:
    if not 1 <= num_threads <= (os.cpu_count() or 1):
        raise ValueError("CPU threads must be between 1 and the available CPU count.")
