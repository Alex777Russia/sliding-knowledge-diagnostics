import json
from typing import Any, Dict, List, Optional, Union

from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    evaluation_score: float
    evaluation_comment: str


@dataclass
class HistoryElement:
    role: str
    content: str
    gt_answer: Optional[str] = None
    blum_level: Optional[str] = None
    evaluation_result: Optional[EvaluationResult] = None
    voice_answer_available: bool = False
    audio_file_path: Optional[str] = None


def smart_json_loads(json_string):
    try:
        start_index = json_string.find('{')
        end_index = json_string.rfind('}')

        if start_index != -1 and end_index != -1:
            clean_json_string = json_string[start_index : end_index + 1]
            data = json.loads(clean_json_string)
            return data
        
        else:
            return {}

    except json.JSONDecodeError as e:
        return {}