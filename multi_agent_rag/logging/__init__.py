import logging
import os
from datetime import datetime

# Main log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs/debug_logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# General logger setup (log to file and console)
logger = logging.getLogger()  # Default logger
logger.setLevel(logging.DEBUG)

# File handler for writing logs to file
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler for terminal output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

# Response logger setup (for different logging purposes)
response_log_path = os.path.join(
    os.getcwd(), "logs/response_logs", f"responses_{LOG_FILE}"
)
os.makedirs(os.path.dirname(response_log_path), exist_ok=True)
response_handler = logging.FileHandler(response_log_path)
response_handler.setLevel(logging.INFO)
response_handler.setFormatter(formatter)

response_logger = logging.getLogger("response_logger")
response_logger.setLevel(logging.INFO)  # You can change this level if needed
response_logger.addHandler(response_handler)
