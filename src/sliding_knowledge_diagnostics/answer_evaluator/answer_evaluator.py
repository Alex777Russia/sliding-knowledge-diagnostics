import json
from typing import Dict, List, Tuple

from src.sliding_knowledge_diagnostics.utils import EvaluationResult, HistoryElement, smart_json_loads
from src.sliding_knowledge_diagnostics.answer_evaluator.prompts import (
    ANSWER_EVALUATOR_SYSTEM_PROMPT, 
    ANSWER_EVALUATOR_USER_TEMPLATE
)
from src.llm import LLM
from src.logging_config import get_logger

logger = get_logger(__name__)


class AnswerEvaluator:
    def __init__(
            self
    ):
        logger.info("Инициализация AnswerEvaluator")
        self.llm = LLM()
        self.system_prompt = ANSWER_EVALUATOR_SYSTEM_PROMPT
        self.user_template = ANSWER_EVALUATOR_USER_TEMPLATE
        logger.info("AnswerEvaluator успешно инициализирован")

    def evaluate_answer(
            self,
            history: List[Dict[str, str]]
    ) -> EvaluationResult:
        logger.info("Начинаем оценку ответа студента")
        question, student_answer, gt_answer, blum_level = self._extract_data_for_evaluation(
            history=history
        )
        
        # logger.debug(f"Вопрос: {question[:100]}...")
        # logger.debug(f"Ответ студента: {student_answer[:100]}...")
        # logger.debug(f"Эталонный ответ: {gt_answer[:100]}...")
        # logger.debug(f"Уровень Блума: {blum_level}")

        
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

        # logger.info("Отправляем запрос к LLM для оценки ответа")
        model_prediction = self.llm.run(
            messages=messages
        )
        logger.debug(f"Получен ответ от LLM: {model_prediction[:200]}...")
        
        try:
            model_prediction = json.loads(model_prediction)
            evaluation_score, evaluation_comment = model_prediction["evaluation_result"], model_prediction["evaluation_comment"]
            logger.info(f"Успешно распарсен JSON ответ. Оценка: {evaluation_score}")
        except Exception as e:
            logger.warning(f"Ошибка парсинга JSON: {e}. Пробуем smart_json_loads")
            try:
               model_prediction = smart_json_loads(model_prediction) 
               evaluation_score, evaluation_comment = model_prediction["evaluation_result"], model_prediction["evaluation_comment"]
               logger.info(f"Успешно распарсен через smart_json_loads. Оценка: {evaluation_score}")
            except Exception as e2:
                logger.error(f"Не удалось распарсить ответ LLM: {e2}. Устанавливаем оценку 0")
                evaluation_score, evaluation_comment = 0, ""

        logger.info(f"Финальная оценка: {evaluation_score}, комментарий: {evaluation_comment[:100]}...")
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
