from configparser import ConfigParser

__config = ConfigParser()
__config.read(__config.read('config.ini'))


RESULT_PATH = __config["PATH"]["result_path"]

RANDOM_STATE = __config["CONSTANTS"]["random_state"]
CHUNK_SIZE = __config["CONSTANTS"]["chunk_size"]