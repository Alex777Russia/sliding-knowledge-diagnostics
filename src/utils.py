from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional, Union
import uuid

from src.messages_utils import Message, parse_data
from src.quality_manager.quality_result import QMResponse
from src.sources import Passage, Chapter, ChapterType, FAQSource, PassageSource
from src.quality_manager.utils import TaskStatus


@dataclass
class StudentRequest:
    task_id: str
    answer: str
