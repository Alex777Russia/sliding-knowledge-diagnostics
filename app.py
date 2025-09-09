import os
import gradio as gr
from typing import Optional, Dict, List, Tuple
from src.sliding_knowledge_diagnostics.sliding_knowledge_diagnostics import SlidingKnowledgeDiagnostics
from src.utils import EXAM_IS_NOT_DONE_MESSAGE, prettify_numbered_text

StartCallback = Tuple[str, List[Dict[str, str]], SlidingKnowledgeDiagnostics]
AnswerCallback = Tuple[str, List[Dict[str, str]], SlidingKnowledgeDiagnostics, float, str]

def start_exam(topic: str, history: List[Dict[str, str]]) -> StartCallback:
    session = SlidingKnowledgeDiagnostics()
    session.get_question_and_add_question_element_to_history()
    hist_elem = session.history[-1]
    first_question = hist_elem.content
    first_message = f"""
        Привет! Я помогу тебе проверить свои знания по теме {topic}, 
        вот мой первый вопрос:
        {first_question}
    """
    history.append({"role": "assistant", "content": prettify_numbered_text(first_message)})
    return f"Экзамен начат по теме: {topic}", history, session

def answer_question(session: Optional[SlidingKnowledgeDiagnostics],
                    answer: str,
                    history: List[Dict[str, str]]) -> AnswerCallback:
    if not session:
        return "", [{"role": "assistant", "content": 'Для начала тебе необходимо начать экзамен!'}], None
    
    session.process_answer(answer)
    user_elem = session.history[-2]
    assist_elem = session.history[-1]
    history.append({"role": "user", "content": user_elem.content})
    history.append({"role": "assistant", "content": prettify_numbered_text(assist_elem.content)})

    return "", history, session, user_elem.evaluation_result.evaluation_score, assist_elem.blum_level

def show_report(session: Optional[SlidingKnowledgeDiagnostics]) -> str:
    if session:
        report = session.get_report()
        if isinstance(report, dict):
            return report["status"]
        return report
    return "Нет активной сессии."

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Скользящая диагностика")

    with gr.Row():
        with gr.Column(scale=1):
            topic = gr.Dropdown(
                label="Выберите тему экзамена",
                choices=["Животные"],
            )
            start_btn = gr.Button("Начать экзамен")
            session_info = gr.Textbox(label="Информация о сессии", interactive=False)
            current_level = gr.Textbox(label="Текущий уровень Блума", value="Знание")
            score = gr.Textbox(label="Текущий score", value="-")
            state = gr.State()

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type="messages", label="Диалог экзамена")
            msg = gr.Textbox(label="Ваш ответ")
            send_btn = gr.Button("Отправить ответ")

    start_btn.click(start_exam, [topic, chatbot], [session_info, chatbot, state])
    send_btn.click(answer_question, [state, msg, chatbot], [msg, chatbot, state, score, current_level])

    with gr.Tab("Отчет"):
        report_box = gr.Textbox(label="Сырой отчет (можно редактировать)", lines=8)
        show_btn = gr.Button("Показать отчет")
        show_btn.click(show_report, [state], [report_box])
        output = gr.Markdown()
        report_box.change(lambda x: x, inputs=report_box, outputs=output)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    demo.launch()