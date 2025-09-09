from typing import Any, Dict
import pandas as pd

from src.sliding_knowledge_diagnostics.answer_evaluator import AnswerEvaluator
from src.sliding_knowledge_diagnostics.clarrifying_question_generator import ClarrifyingQuestionGenerator
from src.sliding_knowledge_diagnostics.report_generator import ReportGenerator
from src.sliding_knowledge_diagnostics.utils import EvaluationResult, HistoryElement
from src.utils import EXAM_IS_DONE_BECAUSE_OF_MISTAKES, EXAM_IS_DONE_MESSAGE, EXAM_IS_NOT_DONE_MESSAGE


BLOOM_ORDER_REVERSED = ["Знание", "Понимание", "Применение", "Анализ", "Синтез", "Оценка"][::-1]


class SlidingKnowledgeDiagnostics:
    def __init__(
            self
    ) -> None:
        self.history = []
        self.questions_df = pd.read_csv('data.csv')
        self._set_next_blum_level()
        self.report = None
        self.available_attempts = 2

        self.answer_evaluator = AnswerEvaluator()
        self.clarrifying_question_generator = ClarrifyingQuestionGenerator()
        self.report_generator = ReportGenerator()

    def get_question_and_add_question_element_to_history(self) -> None:
        row = self.questions_df[
            self.questions_df['blum_level'] == self.current_blum_level
        ].iloc[0]
        self.questions_df = self.questions_df.drop(index=row.name)
        self.history.append(
            self._create_history_elem_from_row(
                role="assistant",
                row=row
            )
        )

    def add_user_element_to_history(
            self, 
            answer: str
        ) -> None:
        self.history.append(
            HistoryElement(
                role="user",
                content=answer
            )
        )

    def process_answer(self, answer: str) -> None:
        if self.report:
            return [
                HistoryElement(
                    role="assistant",
                    content=EXAM_IS_DONE_MESSAGE
                )
            ]

        self.add_user_element_to_history(answer)

        evaluation_result = self.answer_evaluator.evaluate_answer(
            history=self.history
        )
        self._add_evaluation_results_to_history(
            evaluation_result=evaluation_result
        )

        if evaluation_result.evaluation_score < 0.4:
            self._decrease_available_attempts()

            if not self._check_if_attempts_exists():
                self.report = self.report_generator.generate_report(self.history)
                self.history.append(HistoryElement(role="assistant", content=EXAM_IS_DONE_BECAUSE_OF_MISTAKES))
            else:
                self.get_question_and_add_question_element_to_history()
        elif 0.4 <= evaluation_result.evaluation_score < 0.6:
            question_element = self.clarrifying_question_generator.get_clarrifying_question_element(
                history=self.history
            )
            self.history.append(question_element)
        else:
            self._set_next_blum_level()
            self._update_max_passed_level()
            self.get_question_and_add_question_element_to_history()

    def get_report(self):
        if self.report:
            return self.report
        return {"status": EXAM_IS_NOT_DONE_MESSAGE}
    
    @staticmethod
    def _create_history_elem_from_row(
        role: str,
        row: dict
    ) -> Dict[str, str]:
        if row["answer"]:
            gt_answer = f"Ответ: {row['answer']}. Решение: {row['problem']}"
        else:
            gt_answer = row["problem"]

        return HistoryElement(
            role=role,
            content=row["problem"],
            gt_answer=gt_answer,
            blum_level=row["blum_level"]
        )
    
    def _decrease_available_attempts(
            self
    ) -> None:
        self.available_attempts -= 1

    def _check_if_attempts_exists(
            self
    ) -> bool:
        return self.available_attempts > 0

    def _add_evaluation_results_to_history(
            self,
            evaluation_result: EvaluationResult
    ) -> None:
        self.history[-1].evaluation_result = evaluation_result

    def _update_max_passed_level(
            self
    ) -> None:
        self.max_passed_level = self.current_blum_level
    
    def _set_next_blum_level(
            self
    ) -> None:
        try:
            self.current_blum_level = BLOOM_ORDER_REVERSED.pop()
        except IndexError:
            pass
    
