from enum import StrEnum, auto
from typing import Any, Dict, List, Optional, Union

from dataclasses import dataclass, field


class TaskStatus(StrEnum):
    SUCCESS = auto()
    ERROR = auto()

@dataclass
class ErrorTaskResult:
    error: str

@dataclass
class GenerationMetricResponse:
    metric: str
    value: Union[Dict[Any, Any], List[Any], str, int, float]

@dataclass
class GenerationTaskResult:
    metrics: List[GenerationMetricResponse]

@dataclass
class TaskResult:
    task_name: str
    task_status: TaskStatus
    task_result: Union[ErrorTaskResult, GenerationTaskResult]
    id: Optional[str] = None
