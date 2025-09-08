from dataclasses import dataclass, field
from typing import List, Optional

from src.quality_manager.utils import TaskResult, AggregatedMetricResponse, ControlSetReport


@dataclass
class StudentResponse:
    task_responses: List[TaskResult]
    aggregated_task_responses: Optional[List[AggregatedMetricResponse]] = field(default_factory=list)
    control_set_report: Optional[List[ControlSetReport]] = field(default_factory=list)
