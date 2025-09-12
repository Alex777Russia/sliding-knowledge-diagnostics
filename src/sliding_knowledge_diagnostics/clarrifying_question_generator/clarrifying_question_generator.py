import json
from typing import Dict, List, Tuple

from src.sliding_knowledge_diagnostics.utils import HistoryElement, smart_json_loads
from src.sliding_knowledge_diagnostics.clarrifying_question_generator.prompts import (
    CLARRIFYING_QUESTION_GENERATOR_SYSTEM_PROMPT, 
    CLARRIFYING_QUESTION_GENERATOR_USER_TEMPLATE
)
from src.llm import LLM
from src.logging_config import get_logger

logger = get_logger(__name__)


class ClarrifyingQuestionGenerator:
    def __init__(
            self
    ):
        self.llm = LLM()
        self.system_prompt = CLARRIFYING_QUESTION_GENERATOR_SYSTEM_PROMPT
        self.user_template = CLARRIFYING_QUESTION_GENERATOR_USER_TEMPLATE

    def get_clarrifying_question_element(
            self,
            history: List[Dict[str, str]]
    ) -> HistoryElement:
        question, student_answer, gt_answer, blum_level = self._extract_data_for_generation(
            history=history
        )
        
        messages = [
            {
                "role": "system",
                "text": CLARRIFYING_QUESTION_GENERATOR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "text": CLARRIFYING_QUESTION_GENERATOR_USER_TEMPLATE.format(
                    question=question,
                    student_answer=student_answer,
                    gt_answer=gt_answer,
                    blum_level=blum_level
                )
            }
        ]

        model_prediction = self.llm.run(
            messages=messages
        )

        try:
            model_prediction = json.loads(model_prediction)
            question, gt_answer = model_prediction["question"], model_prediction["answer"]
        except Exception as e:
            try:
               model_prediction = smart_json_loads(model_prediction) 
               question, gt_answer = model_prediction["question"], model_prediction["answer"]
            except Exception as e2:
                question, gt_answer = "", ""

        return HistoryElement(
            role="assistant",
            content=question,
            gt_answer=gt_answer,
            blum_level=blum_level
        )

    @staticmethod
    def _extract_data_for_generation(
        history: List[HistoryElement]
    ) -> Tuple[str, str, str, str]:
        question_element, answer_element = history[-2:]

        return (
            question_element.content, 
            answer_element.content, 
            question_element.gt_answer, 
            question_element.blum_level
        )
