import json
from typing import Dict, List

from src.sliding_knowledge_diagnostics.report_generator.prompts import (
    REPORT_GENERATOR_SYSTEM_PROMPT, 
    REPORT_GENERATOR_USER_TEMPLATE
)
from src.sliding_knowledge_diagnostics.utils import HistoryElement
from src.llm import LLM


class ReportGenerator:
    def __init__(
            self
    ):
        self.llm = LLM()
        self.system_prompt = REPORT_GENERATOR_SYSTEM_PROMPT
        self.user_template = REPORT_GENERATOR_USER_TEMPLATE

    def generate_report(
            self,
            history: List[Dict[str, str]]
    ) -> str:
        structured_data = self._extract_data_for_report(
            history=history
        )
        messages = [
            {
                "role": "system",
                "text": REPORT_GENERATOR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "text": REPORT_GENERATOR_USER_TEMPLATE.format(
                    structured_data=structured_data
                )
            }
        ]

        report = self.llm.run(
            messages=messages
        )

        return report
    
    @staticmethod
    def _extract_data_for_report(
        history: List[HistoryElement]
    ) -> str:
        all_data = []
        for i in range(0, len(history) - 1, 2):
            question_element, answer_element = history[i], history[i + 1]
            all_data.append(
                {
                "Вопрос": question_element.content,
                "Ответ студента": answer_element.content,
                "Верный ответ": question_element.gt_answer,
                "Уровень таксономии Блума": question_element.blum_level,
                "Оценка ответа студента": answer_element.evaluation_result.evaluation_score,
                "Комментарий по поводу оценки": answer_element.evaluation_result.evaluation_comment
                }
            )

        return json.dumps(all_data, indent=4, ensure_ascii=False)
