from loguru import logger
from mmengine import Config

config = Config.fromfile("config.py")
logger.info("Config loaded successfully.")
