import logging
from ev_load_fc.config import CFG, resolve_path


def setup_logging(log_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger that writes to a file and console.

    Args:
        log_name (str): Name of the log file (with or without .log extension)
        level (int): Logging level 
                     Choose from:
                        logging.DEBUG    OR 10
                        logging.INFO     OR 20
                        logging.WARNING  OR 30
                        logging.ERROR    OR 40
                        logging.CRITICAL OR 50

    Returns:
        logging.Logger: Configured logger
    """
    log_dir = resolve_path(CFG["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)

    if not log_name.endswith('.log'):
        log_name = log_name + '.log'

    log_file = log_dir / log_name

    # Remove existing handlers (to allow reconfiguring)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger