from typing import Dict, List, Union, Optional, Any

from pympler import asizeof

from src.multiprocessing_handler import multiprocessing_handler
from src.sliding_knowledge.quality_result import StudentResponse
from src.sliding_knowledge.utils import TaskResult
from src.task_executors import GenerationTaskExecutor
from src.utils import StudentRequest


class SlidingKnowledge:
    def __init__(
            self,
            cfg_path: str = "configs/yandex_gpt_config.yaml",
    ) -> None:
        self.generation_task_executor = GenerationTaskExecutor()

    @multiprocessing_handler
    def run_generation_task(
            self,
            request: StudentRequest,
            predict_params: Optional[dict] = None,
            external_uuid: Optional[str] = None
    ) -> TaskResult:
        task_response = self.generation_task_executor.run(
            request=request,
            predict_params=predict_params,
            external_uuid=external_uuid,
        )
        return task_response
