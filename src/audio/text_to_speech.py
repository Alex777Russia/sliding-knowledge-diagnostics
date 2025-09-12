import os
import tempfile
import logging
from typing import Optional
from openai import OpenAI
from src.custom_logger import log_function_call

logger = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
    
    @log_function_call
    def generate_speech(
            self, 
            text: str, 
            voice: str = "alloy",
            model: str = "tts-1",
            output_format: str = "mp3"
    ) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        
        if len(text.strip()) == 0:
            return None
        
        text = text[:4096]

        try:
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=output_format
            )
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            return temp_file_path
            
        except Exception as e:
            return None
    
    def generate_speech_for_question(self, question_text: str) -> Optional[str]:
        clean_text = self._clean_text_for_speech(question_text)
        
        return self.generate_speech(
            text=clean_text,
            voice="nova", 
            model="tts-1-hd",  
            output_format="mp3"
        )
    
    def _clean_text_for_speech(self, text: str) -> str:
        text = text.replace("🎤", "")
        text = text.replace("**", "")  # Убираем жирный текст
        text = text.replace("*", "")   # Убираем курсив
        
        text = " ".join(text.split())
        
        return text.strip()
    
    def cleanup_audio_file(self, file_path: str) -> None:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
