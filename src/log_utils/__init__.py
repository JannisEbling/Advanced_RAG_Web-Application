import logging
import os
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + "%(message)s" + reset,
        logging.INFO: blue + "%(message)s" + reset,
        logging.WARNING: yellow + "⚠️ WARNING: %(message)s" + reset,
        logging.ERROR: red + "🚫 ERROR: %(message)s" + reset,
        logging.CRITICAL: bold_red + "🔥 CRITICAL: %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logging(log_dir: Optional[str] = None) -> tuple[logging.Logger, logging.Logger]:
    """
    Setup logging configuration with both file and console output.
    
    Args:
        log_dir: Optional custom log directory path
        
    Returns:
        Tuple of (main logger, response logger)
    """
    # Create timestamp for log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup log directories
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    else:
        log_dir = Path(log_dir)
        
    debug_log_dir = log_dir / "debug_logs"
    response_log_dir = log_dir / "response_logs"
    
    for directory in [debug_log_dir, response_log_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Setup main logger
    logger = logging.getLogger("rag_pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Detailed formatter for file logs
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler for debug logs
    debug_file = debug_log_dir / f"debug_{timestamp}.log"
    file_handler = logging.FileHandler(debug_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with colored output for warnings and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # Setup response logger
    response_logger = logging.getLogger("response_logger")
    response_logger.setLevel(logging.INFO)
    
    # File handler for response logs
    response_file = response_log_dir / f"responses_{timestamp}.log"
    response_handler = logging.FileHandler(response_file)
    response_handler.setLevel(logging.INFO)
    response_handler.setFormatter(file_formatter)
    response_logger.addHandler(response_handler)
    
    return logger, response_logger


# Initialize loggers
logger, response_logger = setup_logging()