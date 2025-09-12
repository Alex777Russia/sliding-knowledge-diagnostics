import os
import tempfile
import logging
from typing import Optional
from openai import OpenAI


logger = logging.getLogger(__name__)


class AudioTranscriber:
    def __init__(
            self, 
            api_key: Optional[str] = None
    ):
        self.client = OpenAI(api_key=api_key)
    
    def transcribe_audio(
            self, 
            filepath: str, 
            language: str = "ru",
            model: str = "gpt-4o-mini-transcribe"
    ) -> str:
        if not filepath or not isinstance(filepath, str):
            return "Ошибка: неверный путь к файлу"
            
        if not os.path.exists(filepath):
            return f"Ошибка: файл {filepath} не найден"
        
        if os.path.isdir(filepath):
            return f"Ошибка: {filepath} является директорией, а не файлом"
        
        file_size = os.path.getsize(filepath)
        if file_size > 25 * 1024 * 1024:  # 25MB
            return "Ошибка: файл слишком большой (максимум 25MB)"
        
        try:
            with open(filepath, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
            return transcript.strip()
        except Exception as e:
            return f"Ошибка при транскрипции: {str(e)}"
    
    def transcribe_audio_bytes(
            self, 
            audio_bytes: bytes, 
            language: str = "ru",
            model: str = "gpt-4o-mini-transcribe"
    ) -> str:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            result = self.transcribe_audio(temp_file_path, language, model)
            
            os.unlink(temp_file_path)
            
            return result
        except Exception as e:
            return f"Ошибка при транскрипции аудио: {str(e)}"
