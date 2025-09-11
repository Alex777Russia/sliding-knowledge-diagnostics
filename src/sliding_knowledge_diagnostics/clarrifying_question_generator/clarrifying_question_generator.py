import json
from typing import Dict, List, Tuple

from src.sliding_knowledge_diagnostics.utils import EvaluationResult, HistoryElement, smart_json_loads
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
        logger.info("Генерация уточняющего вопроса")
        question, student_answer, gt_answer, blum_level = self._extract_data_for_generation(
            history=history
        )
        
        logger.debug(f"Исходный вопрос: {question[:100]}...")
        logger.debug(f"Ответ студента: {student_answer[:100]}...")
        logger.debug(f"Уровень Блума: {blum_level}")
        
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

        logger.info("Отправляем запрос к LLM для генерации уточняющего вопроса")
        model_prediction = self.llm.run(
            messages=messages
        )
        logger.debug(f"Получен ответ от LLM: {model_prediction[:200]}...")

        try:
            model_prediction = json.loads(model_prediction)
            question, gt_answer = model_prediction["question"], model_prediction["answer"]
            logger.info("Успешно распарсен JSON ответ")
        except Exception as e:
            logger.warning(f"Ошибка парсинга JSON: {e}. Пробуем smart_json_loads")
            try:
               model_prediction = smart_json_loads(model_prediction) 
               question, gt_answer = model_prediction["question"], model_prediction["answer"]
               logger.info("Успешно распарсен через smart_json_loads")
            except Exception as e2:
                logger.error(f"Не удалось распарсить ответ LLM: {e2}. Устанавливаем пустые значения")
                question, gt_answer = "", ""

        logger.info(f"Сгенерирован уточняющий вопрос: {question[:100]}...")
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
