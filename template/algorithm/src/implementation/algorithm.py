from logging import getLogger
from pathlib import Path
from typing import Any, Optional, TypeVar
from oceanprotocol_job_details.ocean import JobDetails
import json

# =============================== IMPORT LIBRARY ====================
# TODO: import library here
# =============================== END ===============================

T = TypeVar("T")

logger = getLogger(__name__)


class Algorithm:
    # TODO: [optional] add class variables here

    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[Any] = None

    def _validate_input(self) -> None:
        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

   
   
    def run(self) -> "Algorithm":
        # 1. Initialize results type
        self.results = {}

        # 2. validate input here
        self._validate_input()

        # 3. get input files here
        input_files = self._job_details.files.files[0].input_files
        filename = str(input_files[0])

        # 4. run algorithm here: count lines
        try:
            with open(filename, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            self.results = {"line_count": line_count}
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            self.results = {"error": str(e)}

        # 5. save results here (handled in save_result)

        # 6. return self
        return self

    def save_result(self, path: Path) -> None:
        # 7. define/add result path here
        result_path = path / "result.json"

        with open(result_path, "w", encoding="utf-8") as f:
            try:
                # 8. save results here
                json.dump(self.results, f, indent=2)
            except Exception as e:
                logger.exception(f"Error saving data: {e}")

       