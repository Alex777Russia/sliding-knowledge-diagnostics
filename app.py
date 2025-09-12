import os
import gradio as gr
from typing import Optional, Dict, List, Tuple
from src.custom_logger import log_function_call
from src.sliding_knowledge_diagnostics.sliding_knowledge_diagnostics import SlidingKnowledgeDiagnostics
from src.utils import (
    CURRENT_BLUM_LEVEL_LABEL, 
    CURRENT_SCORE_LABEL, 
    DOWNLOAD_REPORT_BUTTON, 
    EXAM_DIALOG_LABEL, 
    NO_ACTIVE_SESSION_REPORT, 
    PAGE_TITLE, 
    PICK_TOPIC_LABEL, 
    RAW_REPORT_LABEL, 
    RECOGNIZED_ANSWER, 
    SEND_TEXT_ANSWER_BUTTON, 
    SEND_VOICE_ANSWER_BUTTON, 
    SESSION_INFO_LABEL, 
    SHOW_REPORT_BUTTON, 
    START_EXAM_BUTTON, 
    TEXT_ANSWER_TAB, 
    VOICE_ANSWER_TAB, 
    YOUR_ANSWER_LABEL, 
    prettify_numbered_text
)
from src.audio import AudioTranscriber, VoiceAnalyzer

CURRENT_AVAILABLE_TOPICS = ["Животные"]
DOWNLOAD_PATH = "report.pdf"

StartCallback = Tuple[str, List[Dict[str, str]], SlidingKnowledgeDiagnostics, bool, str]
AnswerCallback = Tuple[str, List[Dict[str, str]], SlidingKnowledgeDiagnostics, float, str, bool, str]
AudioCallback = Tuple[str, str, List[Dict[str, str]], SlidingKnowledgeDiagnostics, float, str, str, bool, str]


try:
    audio_transcriber = AudioTranscriber()
except Exception as e:
    audio_transcriber = None

try:
    voice_analyzer = VoiceAnalyzer()
except Exception as e:
    voice_analyzer = None

@log_function_call
def start_exam(topic: str, history: List[Dict[str, str]]) -> StartCallback:
    session = SlidingKnowledgeDiagnostics()
    session.get_question_and_add_question_element_to_history()
    hist_elem = session.history[-1]
    first_question = hist_elem.content
    
    requires_voice = hist_elem.voice_answer_available if hasattr(hist_elem, 'voice_answer_available') else False
    audio_file_path = hist_elem.audio_file_path if hasattr(hist_elem, 'audio_file_path') else None
    
    first_message = f"""
        Привет! Я помогу тебе проверить свои знания по теме {topic}, 
        вот мой первый вопрос:
        {first_question}
    """
    history = [{"role": "assistant", "content": prettify_numbered_text(first_message)}]
    return f"Экзамен начат по теме: {topic}", history, session, requires_voice, audio_file_path or "", ""

@log_function_call
def answer_question(session: Optional[SlidingKnowledgeDiagnostics],
                    answer: str,
                    history: List[Dict[str, str]]) -> AnswerCallback:
    if not session:
        return (
            "", [{"role": "assistant", "content": 'Для начала тебе необходимо начать экзамен!'}], 
            None, 0.0, "", False, ""
        )
    
    session.process_answer(answer)
    user_elem = session.history[-2]
    assist_elem = session.history[-1]

    requires_voice = False
    audio_file_path = None
    if hasattr(assist_elem, 'voice_answer_available'):
        requires_voice = assist_elem.voice_answer_available
    if hasattr(assist_elem, 'audio_file_path'):
        audio_file_path = assist_elem.audio_file_path

    history.append({"role": "user", "content": user_elem.content})
    history.append({"role": "assistant", "content": prettify_numbered_text(assist_elem.content)})
    return (
        "", history, session, user_elem.evaluation_result.evaluation_score, 
        assist_elem.blum_level, requires_voice, audio_file_path or ""
    )

@log_function_call
def process_audio_answer(session: Optional[SlidingKnowledgeDiagnostics],
                        audio_file: str,
                        history: List[Dict[str, str]]) -> AudioCallback:
    if not session:
        return (
            "", "", [{"role": "assistant", "content": 'Для начала тебе необходимо начать экзамен!'}], 
            None, 0.0, "", None, False, ""
        )
    
    if not audio_file or audio_file is None:
        return (
            "", "", [{"role": "assistant", "content": 'Пожалуйста, запишите аудио ответ!'}], 
            session, 0.0, "", None, False, ""
        )
    
    if not isinstance(audio_file, str) or not os.path.exists(audio_file) or os.path.isdir(audio_file):
        return (
            "", "", [{"role": "assistant", "content": 'Пожалуйста, запишите аудио ответ!'}], 
            session, 0.0, "", None, False, ""
        )
    
    if not audio_transcriber:
        return (
            "", "", [{"role": "assistant", "content": 'Аудио транскрипция недоступна. Проверьте настройки API.'}], 
            session, 0.0, "", None, False, ""
        )
    
    transcribed_text = audio_transcriber.transcribe_audio(audio_file, language="ru")
    
    if not transcribed_text or transcribed_text.startswith("Ошибка"):
        return (
            "", "", [{"role": "assistant", "content": f'Ошибка транскрипции: {transcribed_text}'}], 
            session, 0.0, "", None, False, ""
        )
    
    session.process_answer(transcribed_text, audio_file)
    user_elem = session.history[-2]
    assist_elem = session.history[-1]
    
    requires_voice = False
    audio_file_path = None
    if hasattr(assist_elem, 'voice_answer_available'):
        requires_voice = assist_elem.voice_answer_available
    if hasattr(assist_elem, 'audio_file_path'):
        audio_file_path = assist_elem.audio_file_path
    
    history.append({"role": "user", "content": f"[Аудио] {user_elem.content}"})
    history.append({"role": "assistant", "content": prettify_numbered_text(assist_elem.content)})
    
    return (
        None, transcribed_text, history, session, user_elem.evaluation_result.evaluation_score, 
        assist_elem.blum_level, None, requires_voice, audio_file_path or ""
    )

@log_function_call
def show_report(session: Optional[SlidingKnowledgeDiagnostics]) -> str:
    if session:
        report = session.get_report()
        if isinstance(report, dict):
            return report["status"]
        return report
    return NO_ACTIVE_SESSION_REPORT

with gr.Blocks(theme=gr.themes.Soft(), css="""
.compact-audio audio {
    height: 10px;   /* уменьшенная высота плеера */
    width: 240px;   /* ширина */
    border-radius: 4px;
}
.compact-audio {
    margin: 1px 0;  /* отступы сверху и снизу */
}
""") as demo:
    gr.Markdown(PAGE_TITLE)

    with gr.Row():
        with gr.Column(scale=1):
            topic = gr.Dropdown(
                label=PICK_TOPIC_LABEL,
                choices=CURRENT_AVAILABLE_TOPICS,
            )
            start_btn = gr.Button(START_EXAM_BUTTON)
            session_info = gr.Textbox(label=SESSION_INFO_LABEL, interactive=False)
            current_level = gr.Textbox(label=CURRENT_BLUM_LEVEL_LABEL, value="Знание")
            score = gr.Textbox(label=CURRENT_SCORE_LABEL, value="-")
            state = gr.State()
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type="messages", label=EXAM_DIALOG_LABEL)

            # компактный плеер для озвучки вопроса
            question_audio = gr.Audio(
                label=None,
                interactive=False,
                visible=False,
                show_download_button=False,
                type="filepath",
                elem_classes="compact-audio"
            )

            with gr.Tabs() as tabs:
                with gr.Tab(TEXT_ANSWER_TAB, id="text_tab"):
                    msg = gr.Textbox(label=YOUR_ANSWER_LABEL)
                    send_btn = gr.Button(SEND_TEXT_ANSWER_BUTTON)
                
                with gr.Tab(VOICE_ANSWER_TAB, id="voice_tab"):
                    audio_input = gr.Audio(
                        sources=["microphone"], 
                        type="filepath",
                        value=None
                    )
                    audio_send_btn = gr.Button(SEND_VOICE_ANSWER_BUTTON)
                    transcribed_text = gr.Textbox(
                        label=RECOGNIZED_ANSWER, 
                        interactive=False,
                        visible=True
                    )

    voice_required_state = gr.State(False)
    audio_file_state = gr.State("")
    
    def switch_tab_based_on_voice(requires_voice):
        if requires_voice:
            return gr.update(selected="voice_tab")
        else:
            return gr.update(selected="text_tab")
    
    def update_audio_player(audio_file_path, requires_voice, session_state):
        if requires_voice and audio_file_path and session_state:
            return gr.update(value=audio_file_path, visible=True)  # question_audio
        else:
            return gr.update(visible=False)  # question_audio
    
    with gr.Tab("Отчет"):
        gr.set_static_paths(paths=[DOWNLOAD_PATH])
        report_box = gr.Textbox(label=RAW_REPORT_LABEL, lines=8)
        show_btn = gr.Button(SHOW_REPORT_BUTTON)
        gr.DownloadButton(label=DOWNLOAD_REPORT_BUTTON, value=DOWNLOAD_PATH)
        show_btn.click(show_report, [state], [report_box])
        output = gr.Markdown()
        report_box.change(lambda x: x, inputs=report_box, outputs=output)

    start_btn.click(
        start_exam, 
        [topic, chatbot], 
        [session_info, chatbot, state, voice_required_state, audio_file_state, report_box]
    ).then(
        switch_tab_based_on_voice,
        inputs=[voice_required_state],
        outputs=[tabs]
    ).then(
        update_audio_player,
        inputs=[audio_file_state, voice_required_state, state],
        outputs=[question_audio]
    )

    send_btn.click(
        answer_question, 
        [state, msg, chatbot], 
        [msg, chatbot, state, score, current_level, voice_required_state, audio_file_state]
    ).then(
        switch_tab_based_on_voice,
        inputs=[voice_required_state],
        outputs=[tabs]
    ).then(
        update_audio_player,
        inputs=[audio_file_state, voice_required_state, state],
        outputs=[question_audio]
    )
    
    audio_send_btn.click(
        process_audio_answer, 
        [state, audio_input, chatbot], 
        [
            gr.State(None), 
            transcribed_text, 
            chatbot, state, 
            score, 
            current_level, 
            audio_input, 
            voice_required_state, 
            audio_file_state
        ]
    ).then(
        switch_tab_based_on_voice,
        inputs=[voice_required_state],
        outputs=[tabs]
    ).then(
        update_audio_player,
        inputs=[audio_file_state, voice_required_state, state],
        outputs=[question_audio]
    )
    

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    demo.launch()
