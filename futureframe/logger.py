import logging
import time

from rich.logging import RichHandler


class ElapsedTimeFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%", *args, **kwargs):
        super().__init__(fmt, datefmt, style, *args, **kwargs)
        self.last_log_time = None

    def format(self, record):
        current_time = time.time()
        if self.last_log_time is None:
            elapsed_time = 0
        else:
            elapsed_time = current_time - self.last_log_time
        self.last_log_time = current_time

        # Add elapsed time to the log record
        record.elapsed_time = elapsed_time
        return super().format(record)


# Define the format for the log messages
FORMAT = "%(elapsed_time).2fs\t%(message)s"
DATEFMT = "[%X]"

# Set up the logging configuration
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt=DATEFMT,
    handlers=[RichHandler()],
)

# Get the root logger
logger = logging.getLogger()

# Replace the default formatter with the custom one
for handler in logger.handlers:
    handler.setFormatter(ElapsedTimeFormatter(fmt=FORMAT, datefmt=DATEFMT))

log = logger

if __name__ == "__main__":
    # Example usage
    log = logger
    log.setLevel(logging.INFO)
    log.info("This is the first log message.")
    time.sleep(2)
    log.info("This is the second log message after 2 seconds.")
    time.sleep(1)
    log.info("This is the third log message after 1 second.")
