from copy import copy
from typing import Any, Dict
import pandas as pd

from src.sliding_knowledge_diagnostics.answer_evaluator import AnswerEvaluator
from src.sliding_knowledge_diagnostics.clarrifying_question_generator import ClarrifyingQuestionGenerator
from src.sliding_knowledge_diagnostics.report_generator import ReportGenerator
from src.sliding_knowledge_diagnostics.utils import EvaluationResult, HistoryElement
from src.utils import EXAM_IS_DONE_BECAUSE_OF_MISTAKES, EXAM_IS_DONE_MESSAGE, EXAM_IS_NOT_DONE_MESSAGE, VOICE_INPUT_AVAILABLE_MESSAGE
from src.logging_config import get_logger
from src.audio import TextToSpeech

logger = get_logger(__name__)


BLOOM_ORDER_REVERSED = ["Знание", "Понимание", "Применение", "Анализ", "Синтез", "Оценка"][::-1]


class SlidingKnowledgeDiagnostics:
    def __init__(
            self
    ) -> None:
        self.history = []
        self.bloom_oreder_reversed = copy(BLOOM_ORDER_REVERSED)
        self.questions_df = pd.read_csv('data.csv')
        self._set_next_blum_level_and_get_status(True)
        self.report = None
        self.available_attempts = 2

        self.answer_evaluator = AnswerEvaluator()
        self.clarrifying_question_generator = ClarrifyingQuestionGenerator()
        self.report_generator = ReportGenerator()

        
        # Инициализация TTS для озвучки вопросов
        try:
            self.text_to_speech = TextToSpeech()
            logger.info("TTS генератор успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации TTS: {e}")
            self.text_to_speech = None
        
        # logger.info("SlidingKnowledgeDiagnostics успешно инициализирован")

    def get_question_and_add_question_element_to_history(self) -> None:
        available_questions = self.questions_df[
            self.questions_df['blum_level'] == self.current_blum_level
        ]
        
        if len(available_questions) > 0:
            row = available_questions.iloc[0]
            self.questions_df = self.questions_df.drop(index=row.name)
            self.history.append(
                self._create_history_elem_from_row(
                    role="assistant",
                    row=row
                )
            )

        else:
            if self._set_next_blum_level_and_get_status():
                self.get_question_and_add_question_element_to_history()
            else:
                self._generate_report_and_set_status()
                self.history.append(HistoryElement(role="assistant", content=EXAM_IS_DONE_MESSAGE))
    
    def _set_next_blum_level_and_get_status(
            self,
            is_initial_update: bool = False
    ) -> bool:
        try:
            old_level = self.current_blum_level if not is_initial_update else "Уровень не определен"
            self.current_blum_level = self.bloom_oreder_reversed.pop()
            logger.info(f"Переход к следующему уровню Блума: {old_level} -> {self.current_blum_level}")
            return True
        except IndexError:
            return False
        
    def _generate_report_and_set_status(
            self
    ) -> None:
        self.report = self.report_generator.generate_report(self.history)


    def process_answer(self, answer: str) -> None:
        if self.report:
            return [
                HistoryElement(
                    role="assistant",
                    content=EXAM_IS_DONE_MESSAGE
                )
            ]

        self.add_user_element_to_history(answer, audio_file_path)

        evaluation_result = self.answer_evaluator.evaluate_answer(
            history=self.history
        )
        
        self._add_evaluation_results_to_history(
            evaluation_result=evaluation_result
        )

        if evaluation_result.evaluation_score < 0.4:
            self._decrease_available_attempts()

            if not self._check_if_attempts_exists():
                self._generate_report_and_set_status()
                self.history.append(HistoryElement(role="assistant", content=EXAM_IS_DONE_BECAUSE_OF_MISTAKES))
            else:
                self.get_question_and_add_question_element_to_history()
        elif 0.4 <= evaluation_result.evaluation_score < 0.6:
            question_element = self.clarrifying_question_generator.get_clarrifying_question_element(
                history=self.history
            )
            self.history.append(question_element)
        else:
            self._set_next_blum_level_and_get_status()
            self._update_max_passed_level()
            self.get_question_and_add_question_element_to_history()

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

    def _add_evaluation_results_to_history(
            self,
            evaluation_result: EvaluationResult
    ) -> None:
        self.history[-1].evaluation_result = evaluation_result

    def _decrease_available_attempts(
            self
    ) -> None:
        self.available_attempts -= 1
        logger.info(f"Уменьшено количество попыток. Осталось: {self.available_attempts}")

    def _check_if_attempts_exists(
            self
    ) -> bool:
        has_attempts = self.available_attempts > 0
        return has_attempts

    def _update_max_passed_level(
            self
    ) -> None:
        self.max_passed_level = self.current_blum_level
    
    def get_report(self):
        if self.report:
            return self.report
        return {"status": EXAM_IS_NOT_DONE_MESSAGE}

    def _create_history_elem_from_row(
            self,
            role: str,
            row: dict
    ) -> Dict[str, str]:
        if row["answer"]:
            gt_answer = f"Ответ: {row['answer']}. Решение: {row['problem']}"
        else:
            gt_answer = row["problem"]

        question_text = row["problem"]

        if row.get("voice_answer_available", False):
            question_text += VOICE_INPUT_AVAILABLE_MESSAGE

        # Генерируем озвучку для вопросов, требующих голосового ответа
        audio_file_path = None
        if voice_answer_available and self.text_to_speech:
            try:
                logger.info("Генерируем озвучку для вопроса")
                audio_file_path = self.text_to_speech.generate_speech_for_question(row["problem"])
                if audio_file_path:
                    logger.info(f"Озвучка сгенерирована: {audio_file_path}")
                else:
                    logger.warning("Не удалось сгенерировать озвучку")
            except Exception as e:
                logger.error(f"Ошибка при генерации озвучки: {e}")

        return HistoryElement(
            role=role,
            content=question_text,
            gt_answer=gt_answer,
            blum_level=row["blum_level"],
            voice_answer_available=voice_answer_available,
            audio_file_path=audio_file_path
        )
