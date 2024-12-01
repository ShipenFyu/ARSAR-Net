import logging
from datetime import datetime
import os


def configure_logging(network, epoch, log_dir='logs'):
    save_dir = os.path.join(log_dir, f'logs_{datetime.now().strftime("%Y_%m_%d")}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_filename = os.path.join(save_dir, f'training_{network}_{epoch}_{datetime.now().strftime("%H%M%S")}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logger = logging.getLogger()
    return logger


def log_training_info(logger, message):
    logger.info(f"{datetime.now()}: {message}")