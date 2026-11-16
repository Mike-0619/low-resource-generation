# utils/logger.py
import logging
import sys

def get_logger(name="babylm"):
    """
    Set up a custom logger with timestamps and log levels.
    This is strictly required for readable SLURM log files.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if the function is called multiple times
    if not logger.handlers:
        # Set the default logging level
        logger.setLevel(logging.INFO)
        
        # Create console handler directed to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter: [YYYY-MM-DD HH:MM:SS] [LEVEL] MESSAGE
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Add formatter to the handler, and handler to the logger
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger