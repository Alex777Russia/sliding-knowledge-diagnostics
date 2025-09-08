from typing import Any, Dict, Tuple

from src import logger
from src.task_executor.core import TaskExecutor
from src.time_utils import Timing
from src.utils import (
    StudentRequest
)
from src.validators import ConfigValidator


class Assistant:
    def __init__(
            self, 
            service_cfg_path: str = "configs/yandex_gpt_configuration.yaml", 
            swagger_path: str = "api/api.yaml"
    ):
        self.task_executor = SlidingKnowledge(service_cfg_path)
        self.config_validator = ConfigValidator(swagger_path)

    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        status, message = self.config_validator.validate_agent_config(config)
        if status != 200:
            raise ValueError(message)

    @Timing(logger=logger)
    def predict(
            self,
            data: Dict[str, Any],
            config: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._validate_configuration(config)

        qm_request = QMRequest.from_dict(data)

        qm_response = self.quality_manager.run_single_eval(qm_request, config)

        result_message = qm_response_to_message(qm_response)

        return result_message, {}
