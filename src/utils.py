from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional, Union
import uuid


@dataclass
class StudentRequest:
    task_id: str
    answer: str
