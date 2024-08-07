import os
from datetime import datetime
import logging
import sys
from typing import Any, Dict
import json

from ..models._ot import OT


class Experiment:
    def __init__(
        self,
        exp_name: str,
        log_dir: str = "logs",
    ):
        self.exp_name = exp_name
        self.log_dir = os.path.join(log_dir, exp_name)
        self.cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # setup loggers
        self.logger = self.init_logger()
        
        # setup sinks
        self.record: Dict[str, Dict] = {}

    def init_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"experiment/{self.exp_name.upper()}")
        logger.propagate = False
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.INFO)
        
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setLevel(logging.INFO)
        logger.addHandler(s_handler)

        f_handler = logging.FileHandler(os.path.join(self.log_dir, f"{self.cur_time}.log"), mode='w')
        f_handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", 
                                                 datefmt="%m/%d %H:%M:%S"))
        f_handler.setLevel(logging.DEBUG)
        logger.addHandler(f_handler)

        return logger


    def run(self, model: Dict[int, OT], *args, **kwargs) -> Any:
        pass

    def checkpoint(self):
        log_path = os.path.join(self.log_dir, self.cur_time + ".json")
        with open(log_path, 'w') as f:
            json.dump(self.record_, f)
        self.logger.info("Checkpoint at {log_path}")
    
    @staticmethod
    def load(log_path: str) -> Dict:
        with open(log_path, "r") as f:
            record_ = json.load(f)
        return record_