import json
from typing import Dict, List, Tuple

from src.sliding_knowledge_diagnostics.utils import EvaluationResult, HistoryElement, smart_json_loads
from src.sliding_knowledge_diagnostics.answer_evaluator.prompts import (
    ANSWER_EVALUATOR_SYSTEM_PROMPT, 
    ANSWER_EVALUATOR_USER_TEMPLATE
)
from src.llm import LLM


class AnswerEvaluator:
    def __init__(
            self
    ):
        self.llm = LLM()
        self.system_prompt = ANSWER_EVALUATOR_SYSTEM_PROMPT
        self.user_template = ANSWER_EVALUATOR_USER_TEMPLATE

    def evaluate_answer(
            self,
            history: List[Dict[str, str]]
    ) -> EvaluationResult:
        question, student_answer, gt_answer, blum_level = self._extract_data_for_evaluation(
            history=history
        )
        messages = [
            {
                "role": "system",
                "text": ANSWER_EVALUATOR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "text": ANSWER_EVALUATOR_USER_TEMPLATE.format(
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
            evaluation_score, evaluation_comment = model_prediction["evaluation_result"], model_prediction["evaluation_comment"]
        except:
            try:
               model_prediction = smart_json_loads(model_prediction) 
               evaluation_score, evaluation_comment = model_prediction["evaluation_result"], model_prediction["evaluation_comment"]
            except:
                evaluation_score, evaluation_comment = 0, ""

        return EvaluationResult(
            evaluation_score=evaluation_score,
            evaluation_comment=evaluation_comment
        )
    
    @staticmethod
    def _extract_data_for_evaluation(
        history: List[HistoryElement]
    ) -> Tuple[str, str, str, str]:
        question_element, answer_element = history[-2:]

        return (
            question_element.content, 
            answer_element.content, 
            question_element.gt_answer, 
            question_element.blum_level
        )
