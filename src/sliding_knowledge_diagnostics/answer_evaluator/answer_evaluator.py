import json
from typing import Dict, List, Tuple

from src.custom_logger import log_function_call
from src.sliding_knowledge_diagnostics.utils import EvaluationResult, HistoryElement, smart_json_loads
from src.sliding_knowledge_diagnostics.answer_evaluator.prompts import (
    ANSWER_EVALUATOR_SYSTEM_PROMPT, 
    ANSWER_EVALUATOR_USER_TEMPLATE,
    VOICE_ANALYZER_SYSTEM_PROMPT,
    VOICE_ANALYZER_USER_TEMPLATE
)
from src.llm import LLM


class AnswerEvaluator:
    def __init__(
            self
    ):
        self.llm = LLM()
        self.system_prompt = ANSWER_EVALUATOR_SYSTEM_PROMPT
        self.user_template = ANSWER_EVALUATOR_USER_TEMPLATE

    @log_function_call
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
        except Exception as e:
            try:
               model_prediction = smart_json_loads(model_prediction) 
               evaluation_score, evaluation_comment = model_prediction["evaluation_result"], model_prediction["evaluation_comment"]
            except Exception as e2:
                evaluation_score, evaluation_comment = 0, ""

        return EvaluationResult(
            evaluation_score=evaluation_score or 0,
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
    
    @log_function_call
    def _enhance_voice_analysis_with_evaluation(
            self, 
            voice_analysis: str, 
            evaluation_comment: str, 
            evaluation_score: float
    ) -> str:
        try:                
            messages = [
                {
                    "role": "system",
                    "text": VOICE_ANALYZER_SYSTEM_PROMPT
                },
                {
                    "role": "user", 
                    "text": VOICE_ANALYZER_USER_TEMPLATE.format(
                        evaluation_score=evaluation_score,
                        evaluation_comment=evaluation_comment,
                        voice_analysis=voice_analysis
                    )
                }
            ]
            
            enhanced_analysis = self.llm.run(messages)
            return f"🎚 Анализ характеристик голоса:\n\n{enhanced_analysis}"
            
        except:
            return voice_analysis
