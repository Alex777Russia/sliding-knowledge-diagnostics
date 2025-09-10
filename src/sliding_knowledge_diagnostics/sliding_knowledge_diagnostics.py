from typing import Any, Dict
import pandas as pd

from src.sliding_knowledge_diagnostics.answer_evaluator import AnswerEvaluator
from src.sliding_knowledge_diagnostics.clarrifying_question_generator import ClarrifyingQuestionGenerator
from src.sliding_knowledge_diagnostics.report_generator import ReportGenerator
from src.sliding_knowledge_diagnostics.utils import EvaluationResult, HistoryElement
from src.utils import EXAM_IS_DONE_BECAUSE_OF_MISTAKES, EXAM_IS_DONE_MESSAGE, EXAM_IS_NOT_DONE_MESSAGE
from src.logging_config import get_logger

logger = get_logger(__name__)


BLOOM_ORDER_REVERSED = ["Знание", "Понимание", "Применение", "Анализ", "Синтез", "Оценка"][::-1]


class SlidingKnowledgeDiagnostics:
    def __init__(
            self
    ) -> None:
        # logger.info("Инициализация SlidingKnowledgeDiagnostics")
        self.history = []
        self.questions_df = pd.read_csv('data.csv')
        logger.info(f"Загружено {len(self.questions_df)} вопросов из data.csv")
        self._set_next_blum_level()
        self.report = None
        self.available_attempts = 2
        # logger.info(f"Доступно попыток: {self.available_attempts}")

        self.answer_evaluator = AnswerEvaluator()
        self.clarrifying_question_generator = ClarrifyingQuestionGenerator()
        self.report_generator = ReportGenerator()
        # logger.info("SlidingKnowledgeDiagnostics успешно инициализирован")

    def get_question_and_add_question_element_to_history(self) -> None:
        logger.info(f"Получение вопроса для уровня Блума: {self.current_blum_level}")
        
        # Попытка найти вопрос для текущего уровня Блума
        available_questions = self.questions_df[
            self.questions_df['blum_level'] == self.current_blum_level
        ]
        
        if len(available_questions) > 0:
            # Есть вопросы для текущего уровня
            row = available_questions.iloc[0]
            logger.debug(f"Выбран вопрос: {row['problem'][:100]}...")
            self.questions_df = self.questions_df.drop(index=row.name)
            self.history.append(
                self._create_history_elem_from_row(
                    role="assistant",
                    row=row
                )
            )
            logger.info(f"Вопрос добавлен в историю. Осталось вопросов: {len(self.questions_df)}")
        else:
            # Нет вопросов для текущего уровня, пытаемся взять следующий уровень
            logger.warning(f"Нет вопросов для уровня {self.current_blum_level}. Пытаемся взять следующий уровень.")
            if self._try_get_next_blum_level():
                # Рекурсивно вызываем метод для нового уровня
                self.get_question_and_add_question_element_to_history()
            else:
                # Не удалось найти подходящий уровень, завершаем экзамен
                logger.error("Не удалось найти подходящие вопросы. Завершаем экзамен.")
                self.report = self.report_generator.generate_report(self.history)
                self.history.append(HistoryElement(role="assistant", content=EXAM_IS_DONE_BECAUSE_OF_MISTAKES))


    def add_user_element_to_history(
            self, 
            answer: str
        ) -> None:
        logger.debug(f"Добавление ответа пользователя в историю: {answer[:100]}...")
        self.history.append(
            HistoryElement(
                role="user",
                content=answer
            )
        )

    def process_answer(self, answer: str) -> None:
        logger.info(f"Обработка ответа пользователя.")
        
        if self.report:
            logger.warning("Попытка обработать ответ после завершения экзамена")
            return [
                HistoryElement(
                    role="assistant",
                    content=EXAM_IS_DONE_MESSAGE
                )
            ]

        self.add_user_element_to_history(answer)

        logger.info("Начинаем оценку ответа")
        evaluation_result = self.answer_evaluator.evaluate_answer(
            history=self.history
        )
        logger.info(f"Оценка получена: {evaluation_result.evaluation_score:.3f}")
        logger.debug(f"Комментарий к оценке: {evaluation_result.evaluation_comment}")
        
        self._add_evaluation_results_to_history(
            evaluation_result=evaluation_result
        )

        if evaluation_result.evaluation_score < 0.4:
            logger.warning(f"Низкая оценка ответа: {evaluation_result.evaluation_score:.3f}")
            self._decrease_available_attempts()

            if not self._check_if_attempts_exists():
                logger.info("Исчерпаны все попытки. Генерируем отчет")
                self.report = self.report_generator.generate_report(self.history)
                self.history.append(HistoryElement(role="assistant", content=EXAM_IS_DONE_BECAUSE_OF_MISTAKES))
            else:
                logger.info(f"Осталось попыток: {self.available_attempts}. Переходим к следующему вопросу")
                self.get_question_and_add_question_element_to_history()
        elif 0.4 <= evaluation_result.evaluation_score < 0.6:
            logger.info(f"Средняя оценка ответа: {evaluation_result.evaluation_score:.3f}. Генерируем уточняющий вопрос")
            question_element = self.clarrifying_question_generator.get_clarrifying_question_element(
                history=self.history
            )
            self.history.append(question_element)
        else:
            logger.info(f"Высокая оценка ответа: {evaluation_result.evaluation_score:.3f}. Переходим к следующему уровню")
            self._set_next_blum_level()
            self._update_max_passed_level()
            self.get_question_and_add_question_element_to_history()

    def get_report(self):
        # logger.info("Запрос отчета")
        if self.report:
            # logger.info("Возвращаем сгенерированный отчет")
            return self.report
        # logger.info("Экзамен еще не завершен")
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

        # Формируем текст вопроса
        question_text = row["problem"]

        # Если доступен голосовой ответ, добавляем просьбу ответить голосом
        if row.get("voice_answer_available", False):
            question_text += "\n\n🎤 **Пожалуйста, ответьте на этот вопрос голосом, используя вкладку 'Голосовой ответ'.**"

        return HistoryElement(
            role=role,
            content=question_text,
            gt_answer=gt_answer,
            blum_level=row["blum_level"],
            voice_answer_available=row.get("voice_answer_available", False)
        )
    
    def _decrease_available_attempts(
            self
    ) -> None:
        self.available_attempts -= 1
        logger.info(f"Уменьшено количество попыток. Осталось: {self.available_attempts}")

    def _check_if_attempts_exists(
            self
    ) -> bool:
        has_attempts = self.available_attempts > 0
        # logger.debug(f"Проверка наличия попыток: {has_attempts}")
        return has_attempts

    def _add_evaluation_results_to_history(
            self,
            evaluation_result: EvaluationResult
    ) -> None:
        # logger.debug("Добавление результатов оценки в историю")
        self.history[-1].evaluation_result = evaluation_result

    def _update_max_passed_level(
            self
    ) -> None:
        # logger.info(f"Обновление максимального пройденного уровня: {self.current_blum_level}")
        self.max_passed_level = self.current_blum_level
    
    def _set_next_blum_level(
            self
    ) -> None:
        try:
            old_level = self.current_blum_level if hasattr(self, 'current_blum_level') else "Не установлен"
            self.current_blum_level = BLOOM_ORDER_REVERSED.pop()
            logger.info(f"Переход к следующему уровню Блума: {old_level} -> {self.current_blum_level}")
        except IndexError:
            logger.warning("Достигнут максимальный уровень Блума или уровни закончились")
            pass
    
    def _try_get_next_blum_level(
            self
    ) -> bool:
        """
        Пытается взять следующий уровень Блума и проверить, есть ли для него вопросы.
        Возвращает True, если удалось найти подходящий уровень с вопросами.
        """
        # Создаем копию списка уровней для проверки
        available_levels = BLOOM_ORDER_REVERSED.copy()
        
        while available_levels:
            try:
                next_level = available_levels.pop(0)
                logger.info(f"Проверяем уровень Блума: {next_level}")
                
                # Проверяем, есть ли вопросы для этого уровня
                questions_for_level = self.questions_df[
                    self.questions_df['blum_level'] == next_level
                ]
                
                if len(questions_for_level) > 0:
                    # Найден уровень с вопросами
                    old_level = self.current_blum_level
                    self.current_blum_level = next_level
                    logger.info(f"Найден подходящий уровень Блума: {old_level} -> {self.current_blum_level}")
                    return True
                else:
                    logger.debug(f"Нет вопросов для уровня {next_level}, пробуем следующий")
                    
            except IndexError:
                logger.warning("Все уровни Блума проверены, подходящих не найдено")
                break
        
        logger.error("Не удалось найти ни одного уровня Блума с доступными вопросами")
        return False
    
