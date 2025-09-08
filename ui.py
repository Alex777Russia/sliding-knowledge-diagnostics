import os
import random
import gradio as gr
import pandas as pd

for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY",
            "ALL_PROXY", "all_proxy"]:
    os.environ.pop(var, None)

BLOOM_ORDER = ["Знание", "Понимание", "Применение", "Анализ", "Синтез", "Оценка"]

def evaluate_answer(student_answer: str, 
                    problem: str, 
                    reference_answer: str,
                    blum_level: str) -> float:
    return random.random()

def generate_clarifying_question(student_answer: str, 
                                 problem: str, 
                                 reference_answer: str,
                                 blum_level: str) -> str:
    return f"Уточняющий вопрос"

def generate_report(history, max_level, topic) -> dict:
    return {
        "topic": topic,
        "max_passed_level": max_level,
        "history": history,
        "summary": ""
    }

def next_bloom_level(current):
    if current in BLOOM_ORDER:
        idx = BLOOM_ORDER.index(current)
        if idx + 1 < len(BLOOM_ORDER):
            return BLOOM_ORDER[idx+1]
    return current


class ExamSession:
    def __init__(self, topic: str):
        self.topic = topic
        self.current_level = "Знание"
        self.max_passed_level = None
        self.history = []
        self.low_score_counter = 0
        self.report = None
        self.questions_df = pd.read_csv('data.csv')
        self.current_row = None

    def get_question(self) -> str:
        row = self.questions_df[
            self.questions_df['blum_level'] == self.current_level
        ].iloc[0]
        # self.questions_df = self.questions_df.drop(index=row.name)
        self.current_row = row
        return row['problem']

    def add_answer(self, question: str, answer: str):
        self.history.append((f"Вопрос ({self.current_level})", answer))

    def process_answer(self, answer: str) -> tuple[float, list[dict]]:
        if self.report:
            txt = "Экзамен завершен! Если хочешь начать новый, нажми кнопку 'Начать экзамен'"
            return [{"role": "assistant", "content": txt}]

        self.history.append({"role": "user", "content": answer})
        score = evaluate_answer(
            answer,
            self.current_row['problem'], 
            self.current_level['answer'],
            self.current_level,
        )

        if score < 0.4:
            self.low_score_counter += 1
            if self.low_score_counter >= 2:
                self.report = generate_report(self.history, self.max_passed_level or self.current_level, self.topic)
                self.history.append({"role": "assistant", "content": "Спасибо, экзамен завершен!"})
            else:
                new_question = self.get_question()
                self.history.append({"role": "assistant", "content": new_question})
        elif 0.4 <= score < 0.6:
            clar_q = generate_clarifying_question(
                answer,
                self.current_row['problem'], 
                self.current_level['answer'],
                self.current_level,
            )
            self.history.append({"role": "assistant", "content": clar_q})
        else:
            self.max_passed_level = self.current_level
            self.current_level = next_bloom_level(self.current_level)
            new_question = self.get_question()
            self.history.append({"role": "assistant", "content": new_question})

        return score, self.history

    def get_report(self):
        if self.report:
            return self.report
        return {"status": "Экзамен ещё не завершён."}


def start_exam(topic: str, history: list[dict]) -> tuple[str, list[dict], ExamSession]:
    session = ExamSession(topic)
    first_question = session.get_question()
    first_message = f"""
        Привет! Я помогу тебе проверить свои знания по теме {topic}, 
        вот мой первый вопрос:
        {first_question}
    """
    session.history.append({"role": "assistant", "content": first_message})
    return f"Экзамен начат по теме: {topic}", session.history, session

def answer_question(answer: str, 
                    session: ExamSession
                   ) -> tuple[str, list[dict], ExamSession, float, str]:
    if not session:
        return "", [{"role": "assistant", "content": 'Для начала тебе необходимо начать экзамен!'}], None
    
    score, new_history = session.process_answer(answer)
    return "", new_history, session, round(score, 2), session.current_level

def show_report(session: ExamSession):
    if session:
        return str(session.get_report())
    return "Нет активной сессии."

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Скользящая диагностика")

    with gr.Row():
        with gr.Column(scale=1):
            topic = gr.Dropdown(
                label="Выберите тему экзамена",
                choices=["Человек", "Организм"],
            )
            start_btn = gr.Button("Начать экзамен")
            session_info = gr.Textbox(label="Информация о сессии", interactive=False)
            current_level = gr.Textbox(label="Текущий уровень Блума", value="-")
            score = gr.Textbox(label="Текущий score", value="-")
            state = gr.State()

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type="messages", label="Диалог экзамена")
            msg = gr.Textbox(label="Ваш ответ")
            send_btn = gr.Button("Отправить ответ")

    start_btn.click(start_exam, [topic, chatbot], [session_info, chatbot, state])
    send_btn.click(answer_question, [msg, state], [msg, chatbot, state, score, current_level])

    with gr.Tab("Отчет"):
        report_box = gr.Textbox(label="Отчет JSON", lines=8)
        show_btn = gr.Button("Показать отчет")
        show_btn.click(show_report, [state], [report_box])

if __name__ == "__main__":
    demo.launch()
