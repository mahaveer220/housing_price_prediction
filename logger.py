import logging
import os
from datetime import datetime
import inspect

class Logger:
    LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
    
    def __init__(self, log_level=logging.INFO):
        """
        Initialize the Logger class.
        :param log_level: Logging level (default: INFO).
        """
        self.log_level = log_level
        
        # Ensure log directory exists
        os.makedirs(self.LOG_DIR, exist_ok=True)
        
        # Get log file for the current day
        self.log_file = os.path.join(self.LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
        
        # Create logger
        self.logger = logging.getLogger("CustomLogger")
        self.logger.setLevel(self.log_level)
        
        # Remove existing handlers to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # File Handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # Formatter
        log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _get_caller_filename(self):
        frame = inspect.stack()[2]
        return os.path.basename(frame.filename)
    
    def info(self, message):
        self.logger.info(f"[{self._get_caller_filename()}] {message}")
    
    def warning(self, message):
        self.logger.warning(f"[{self._get_caller_filename()}] {message}")
    
    def error(self, message):
        self.logger.error(f"[{self._get_caller_filename()}] {message}")


if __name__ == "__main__":
    log = Logger()
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")
