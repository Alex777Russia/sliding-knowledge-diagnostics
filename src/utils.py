from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional, Union
import uuid


EXAM_IS_DONE_MESSAGE = "Экзамен завершен! Если хочешь начать новый, нажми кнопку 'Начать экзамен'"
EXAM_IS_DONE_BECAUSE_OF_MISTAKES = "Спасибо, экзамен завершен!"
EXAM_IS_NOT_DONE_MESSAGE = "Экзамен ещё не завершён."

@dataclass
class StudentRequest:
    task_id: str
    answer: str
