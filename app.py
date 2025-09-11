import os
import gradio as gr
from typing import Optional, Dict, List, Tuple
from src.sliding_knowledge_diagnostics.sliding_knowledge_diagnostics import SlidingKnowledgeDiagnostics
from src.utils import EXAM_IS_NOT_DONE_MESSAGE, prettify_numbered_text
from src.audio import AudioTranscriber, VoiceAnalyzer
from src.logging_config import get_logger

logger = get_logger(__name__)

StartCallback = Tuple[str, List[Dict[str, str]], SlidingKnowledgeDiagnostics, bool, str]
AnswerCallback = Tuple[str, List[Dict[str, str]], SlidingKnowledgeDiagnostics, float, str, bool, str]
AudioCallback = Tuple[str, str, List[Dict[str, str]], SlidingKnowledgeDiagnostics, float, str, str, bool, str]

# Инициализация аудио транскриптера
try:
    logger.info("Инициализация аудио транскриптера...")
    audio_transcriber = AudioTranscriber()
    logger.info("Аудио транскриптер успешно инициализирован")
except Exception as e:
    logger.error(f"Ошибка инициализации аудио транскриптера: {e}")
    audio_transcriber = None

# Инициализация анализатора голоса
try:
    logger.info("Инициализация анализатора голоса...")
    voice_analyzer = VoiceAnalyzer()
    logger.info("Анализатор голоса успешно инициализирован")
except Exception as e:
    logger.error(f"Ошибка инициализации анализатора голоса: {e}")
    voice_analyzer = None

def start_exam(topic: str, history: List[Dict[str, str]]) -> StartCallback:
    logger.info(f"Начало экзамена по теме: {topic}")
    session = SlidingKnowledgeDiagnostics()
    session.get_question_and_add_question_element_to_history()
    hist_elem = session.history[-1]
    first_question = hist_elem.content
    
    # Проверяем, требует ли вопрос голосового ответа
    requires_voice = hist_elem.voice_answer_available if hasattr(hist_elem, 'voice_answer_available') else False
    audio_file_path = hist_elem.audio_file_path if hasattr(hist_elem, 'audio_file_path') else None
    
    first_message = f"""
        Привет! Я помогу тебе проверить свои знания по теме {topic}, 
        вот мой первый вопрос:
        {first_question}
    """
    history.append({"role": "assistant", "content": prettify_numbered_text(first_message)})
    logger.info(f"Экзамен успешно начат. Требуется голосовой ответ: {requires_voice}, Аудио файл: {audio_file_path}")
    return f"Экзамен начат по теме: {topic}", history, session, requires_voice, audio_file_path or ""

def answer_question(session: Optional[SlidingKnowledgeDiagnostics],
                    answer: str,
                    history: List[Dict[str, str]]) -> AnswerCallback:
    logger.info(f"Обработка текстового ответа: {answer[:50]}...")
    if not session:
        logger.warning("Попытка ответить без активной сессии")
        return "", [{"role": "assistant", "content": 'Для начала тебе необходимо начать экзамен!'}], None, 0.0, "", False, ""
    
    session.process_answer(answer)
    user_elem = session.history[-2]
    assist_elem = session.history[-1]

    print('assist_elem')
    print(assist_elem)

    # Проверяем, требует ли следующий вопрос голосового ответа
    requires_voice = False
    audio_file_path = None
    if hasattr(assist_elem, 'voice_answer_available'):
        requires_voice = assist_elem.voice_answer_available
    if hasattr(assist_elem, 'audio_file_path'):
        audio_file_path = assist_elem.audio_file_path

    history.append({"role": "user", "content": user_elem.content})
    history.append({"role": "assistant", "content": prettify_numbered_text(assist_elem.content)})
    logger.info(f"Ответ обработан. Score: {user_elem.evaluation_result.evaluation_score}, Уровень: {assist_elem.blum_level}, Требуется голос: {requires_voice}, Аудио: {audio_file_path}")
    return "", history, session, user_elem.evaluation_result.evaluation_score, assist_elem.blum_level, requires_voice, audio_file_path or ""

def process_audio_answer(session: Optional[SlidingKnowledgeDiagnostics],
                        audio_file: str,
                        history: List[Dict[str, str]]) -> AudioCallback:
    """
    Обработка аудио ответа: транскрипция + обработка ответа
    """
    logger.info(f"Обработка аудио ответа. Файл: {audio_file}")
    
    if not session:
        logger.warning("Попытка аудио ответа без активной сессии")
        return "", "", [{"role": "assistant", "content": 'Для начала тебе необходимо начать экзамен!'}], None, 0.0, "", None, False, ""
    
    # Проверяем, что аудио файл предоставлен и корректен
    if not audio_file or audio_file is None:
        logger.warning("Аудио файл не предоставлен")
        return "", "", [{"role": "assistant", "content": 'Пожалуйста, запишите аудио ответ!'}], session, 0.0, "", None, False, ""
    
    # Дополнительные проверки
    if not isinstance(audio_file, str) or not os.path.exists(audio_file) or os.path.isdir(audio_file):
        logger.warning(f"Некорректный аудио файл: {audio_file}")
        return "", "", [{"role": "assistant", "content": 'Пожалуйста, запишите аудио ответ!'}], session, 0.0, "", None, False, ""
    
    # Проверяем, что транскриптер доступен
    if not audio_transcriber:
        logger.error("Аудио транскриптер не инициализирован")
        return "", "", [{"role": "assistant", "content": 'Аудио транскрипция недоступна. Проверьте настройки API.'}], session, 0.0, "", None, False, ""
    
    # Транскрибируем аудио
    logger.info("Начинаем транскрипцию аудио...")
    transcribed_text = audio_transcriber.transcribe_audio(audio_file, language="ru")
    logger.info(f"Транскрипция завершена: {transcribed_text[:100]}...")
    
    if not transcribed_text or transcribed_text.startswith("Ошибка"):
        logger.error(f"Ошибка транскрипции: {transcribed_text}")
        return "", "", [{"role": "assistant", "content": f'Ошибка транскрипции: {transcribed_text}'}], session, 0.0, "", None, False, ""
    
    # Обрабатываем транскрибированный текст как обычный ответ
    logger.info("Обрабатываем транскрибированный текст...")
    logger.info(f"Транскрибированный текст: {transcribed_text}")
    session.process_answer(transcribed_text, audio_file)
    user_elem = session.history[-2]
    assist_elem = session.history[-1]
    
    # Проверяем, требует ли следующий вопрос голосового ответа
    requires_voice = False
    audio_file_path = None
    if hasattr(assist_elem, 'voice_answer_available'):
        requires_voice = assist_elem.voice_answer_available
    if hasattr(assist_elem, 'audio_file_path'):
        audio_file_path = assist_elem.audio_file_path
    
    history.append({"role": "user", "content": f"[Аудио] {user_elem.content}"})
    history.append({"role": "assistant", "content": prettify_numbered_text(assist_elem.content)})
    
    logger.info(f"Аудио ответ обработан. Score: {user_elem.evaluation_result.evaluation_score}, Уровень: {assist_elem.blum_level}, Требуется голос: {requires_voice}, Аудио: {audio_file_path}")
    return None, transcribed_text, history, session, user_elem.evaluation_result.evaluation_score, assist_elem.blum_level, None, requires_voice, audio_file_path or ""

def show_report(session: Optional[SlidingKnowledgeDiagnostics]) -> str:
    if session:
        report = session.get_report()
        if isinstance(report, dict):
            return report["status"]
        return report
    return "Нет активной сессии."

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
                with gr.Tab("Текстовый ответ", id="text_tab"):
                    msg = gr.Textbox(label="Ваш ответ")
                    send_btn = gr.Button("Отправить ответ")
                
                with gr.Tab("Голосовой ответ", id="voice_tab"):
                    audio_input = gr.Audio(
                        sources=["microphone"], 
                        type="filepath",
                        # label="Запишите или загрузите аудио ответ",
                        value=None
                    )
                    audio_send_btn = gr.Button("Отправить аудио ответ")
                    transcribed_text = gr.Textbox(
                        label="Распознанный текст", 
                        interactive=False,
                        visible=True
                    )

    # Создаем состояние для отслеживания необходимости голосового ответа
    voice_required_state = gr.State(False)
    audio_file_state = gr.State("")
    
    # Функция для переключения вкладок
    def switch_tab_based_on_voice(requires_voice):
        if requires_voice:
            return gr.update(selected="voice_tab")
        else:
            return gr.update(selected="text_tab")
    
    # Функция для обновления аудио плеера
    def update_audio_player(audio_file_path, requires_voice, session_state):
        if requires_voice and audio_file_path and session_state:
            return gr.update(value=audio_file_path, visible=True)  # question_audio
        else:
            return gr.update(visible=False)  # question_audio
    
    # Обработчики событий с автоматическим переключением вкладок
    start_btn.click(
        start_exam, 
        [topic, chatbot], 
        [session_info, chatbot, state, voice_required_state, audio_file_state]
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
        [gr.State(None), transcribed_text, chatbot, state, score, current_level, audio_input, voice_required_state, audio_file_state]
    ).then(
        switch_tab_based_on_voice,
        inputs=[voice_required_state],
        outputs=[tabs]
    ).then(
        update_audio_player,
        inputs=[audio_file_state, voice_required_state, state],
        outputs=[question_audio]
    )

    with gr.Tab("Отчет"):
        report_box = gr.Textbox(label="Сырой отчет (можно редактировать)", lines=8)
        show_btn = gr.Button("Показать отчет")
        show_btn.click(show_report, [state], [report_box])
        output = gr.Markdown()
        report_box.change(lambda x: x, inputs=report_box, outputs=output)


if __name__ == "__main__":
    logger.info("Запуск приложения...")
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Переменные окружения загружены")
    logger.info("Запуск Gradio интерфейса...")
    demo.launch()