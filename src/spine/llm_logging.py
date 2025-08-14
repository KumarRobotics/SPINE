import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional


def get_logger(
    name: str, level: int, stdout: Optional[bool] = True, fpath: Optional[str] = ""
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level=level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if stdout:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if fpath != "":
        fh = logging.FileHandler(filename=fpath)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class LLMDataLogger:
    """Logs llm prompts and queries per inference"""

    def __init__(self, name: str):
        self.file = Path(name)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.touch(exist_ok=True)

    def log(self, msgs: List[Dict[str, str]]):
        # with open(str(self.file), "r+") as f:
        #     json.dump(msgs, f)

        #     def log(self, msgs: List[Dict[str, str]]):

        with open(str(self.file), "r+") as f:
            f.write("[\n")
            for i, msg in enumerate(msgs):
                json_str = json.dumps(msg, indent=2)
                f.write(json_str)
                if i != len(msgs) - 1:
                    f.write(",\n")  # comma + extra newline between entries
                else:
                    f.write("\n")
            f.write("]\n")
