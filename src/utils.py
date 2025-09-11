from dataclasses import dataclass
import re

EXAM_IS_DONE_MESSAGE = "Экзамен завершен! Если хочешь начать новый, нажми кнопку 'Начать экзамен'"
EXAM_IS_DONE_BECAUSE_OF_MISTAKES = "Спасибо, экзамен завершен!"
EXAM_IS_NOT_DONE_MESSAGE = "Экзамен ещё не завершён."

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