from dataclasses import dataclass
import re

EXAM_IS_DONE_MESSAGE = "Экзамен завершен! Если хочешь начать новый, нажми кнопку 'Начать экзамен'"
EXAM_IS_DONE_BECAUSE_OF_MISTAKES = "Спасибо, экзамен завершен!"
EXAM_IS_NOT_DONE_MESSAGE = "Экзамен ещё не завершён."
VOICE_INPUT_AVAILABLE_MESSAGE = "\n\n🎤 **Пожалуйста, ответьте на этот вопрос голосом, используя вкладку 'Голосовой ответ'.**"

NO_ACTIVE_SESSION_REPORT = "Нет активной сессии."

START_EXAM_BUTTON = "Начать экзамен"
SEND_TEXT_ANSWER_BUTTON = "Отправить ответ"
SEND_VOICE_ANSWER_BUTTON = "Отправить аудио ответ"
SHOW_REPORT_BUTTON = "Показать отчет"
DOWNLOAD_REPORT_BUTTON = "Скачать отчет"

PICK_TOPIC_LABEL = "Выберите тему экзамена"
SESSION_INFO_LABEL = "Информация о сессии"
CURRENT_BLUM_LEVEL_LABEL = "Текущий уровень Блума"
CURRENT_SCORE_LABEL = "Текущий score"
EXAM_DIALOG_LABEL = "Диалог экзамена"
YOUR_ANSWER_LABEL = "Ваш ответ"
RECOGNIZED_ANSWER = "Распознанный текст"
RAW_REPORT_LABEL = "Сырой отчет (можно редактировать)"

TEXT_ANSWER_TAB = "Текстовый ответ"
VOICE_ANSWER_TAB = "Голосовой ответ"

PAGE_TITLE = "# Скользящая диагностика"

@dataclass
class StudentRequest:
    task_id: str
    answer: str

def prettify_numbered_text(text: str) -> str:
    parts = re.split(r'(\d+\))', text)
    result = []
    buffer = ""

    for part in parts:
        if re.match(r'\d+\)', part):
            if buffer.strip():
                result.append(buffer.strip())
                buffer = ""
            buffer += f"{part} "
        else:
            buffer += part.strip() + " "

    if buffer.strip():
        result.append(buffer.strip())

    return "\n".join(result)