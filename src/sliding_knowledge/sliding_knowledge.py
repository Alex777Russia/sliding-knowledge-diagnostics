import copy
from multiprocessing.pool import ThreadPool
from omegaconf import DictConfig, OmegaConf
import operator
import os
from typing import Dict, List, Union, Optional, Any

from langchain.schema.embeddings import Embeddings
from pympler import asizeof

from src.embedders import init_embedder
from src.multiprocessing_handler import multiprocessing_handler, thread_context
from src.sliding_knowledge.quality_result import StudentResponse
from src.sliding_knowledge.utils import TaskResult
from src.task_executors import GenerationTaskExecutor
from src.tokenizer.llm_tokenizer import LLMTokenizer
from src.utils import StudentRequest


class SlidingKnowledge:
    def __init__(
            self,
            cfg_path: str = "configs/yandex_gpt_config.yaml",
    ) -> None:
        cfg = OmegaConf.load(cfg_path)
        self.max_task_threads = int(os.environ.get("MAX_TASK_THREADS", 1))
        self.max_ragas_task_threads = int(os.environ.get("RAGAS_MAX_TASK_THREADS", 1))
        self.embedders: Dict[str, Embeddings] = {
            name: init_embedder(
                embedder_parameters=cfg["clients"]["embedder"][name],
                model_name=name,
                tokenizers_cfg=cfg["tokenizer"],
                llms_config=cfg["clients"]["llms"],
            )
            for name in cfg["clients"]["embedder"]
        }
        self.tokenizer = LLMTokenizer(cfg["tokenizer"])
        self.generation_task_executor = GenerationTaskExecutor(
            cfg=dict(cfg["generation_task_executor"]) | {"endpoints": cfg["clients"]["llms"]},
            tokenizer=self.tokenizer,
        )

    def run_generation_task(
            self,
            qm_request: StudentRequest,
            predict_params: Optional[dict] = None,
            external_uuid: Optional[str] = None,
    ) -> TaskResult:
        task_response = self.generation_task_executor.run(
            qm_request=qm_request,
            predict_params=predict_params,
            external_uuid=external_uuid,
        )
        return task_response

    @multiprocessing_handler
    def run_task(
            self,
            request: StudentRequest,
            predict_params: Optional[dict] = None,
            external_uuid: Optional[str] = None,
    ) -> TaskResult:
        task_type_mapping = {
            "universal_generation": self.run_generation_task
        }

        task_function = task_type_mapping.get(task["type"])
        if task_function is None:
            raise ValueError(f"Тип задания {task['type']} не поддерживается")

        return task_function(
            qm_request=qm_request,
            predict_params=predict_params,
            external_uuid=external_uuid,
        )
