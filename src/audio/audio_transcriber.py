import os
import tempfile
import logging
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Класс для транскрипции аудио в текст с использованием OpenAI API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация транскриптера
        
        Args:
            api_key: API ключ OpenAI (если не указан, будет взят из переменной окружения OPENAI_API_KEY)
        """
        logger.info("Инициализация OpenAI клиента для транскрипции аудио")
        self.client = OpenAI(api_key=api_key)
        logger.info("OpenAI клиент успешно инициализирован")
    
    def transcribe_audio(self, 
                        filepath: str, 
                        language: str = "ru",
                        model: str = "gpt-4o-mini-transcribe") -> str:
        """
        Транскрипция аудио файла в текст с использованием OpenAI API
        
        Args:
            filepath: Путь к аудио файлу
            language: Язык для транскрипции (код языка, например "ru" для русского)
            model: Модель для транскрипции
            
        Returns:
            Транскрибированный текст
        """
        logger.info(f"Начинаем транскрипцию файла: {filepath}")
        
        if not filepath or not isinstance(filepath, str):
            logger.error("Неверный путь к файлу")
            return "Ошибка: неверный путь к файлу"
            
        if not os.path.exists(filepath):
            logger.error(f"Файл не найден: {filepath}")
            return f"Ошибка: файл {filepath} не найден"
        
        if os.path.isdir(filepath):
            logger.error(f"Путь ведет к директории: {filepath}")
            return f"Ошибка: {filepath} является директорией, а не файлом"
        
        # Проверяем размер файла (OpenAI имеет ограничение в 25MB)
        file_size = os.path.getsize(filepath)
        logger.info(f"Размер файла: {file_size / 1024 / 1024:.2f} MB")
        if file_size > 25 * 1024 * 1024:  # 25MB
            logger.error(f"Файл слишком большой: {file_size / 1024 / 1024:.2f} MB")
            return "Ошибка: файл слишком большой (максимум 25MB)"
        
        try:
            logger.info(f"Отправляем запрос в OpenAI API (модель: {model}, язык: {language})")
            with open(filepath, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
            logger.info(f"Транскрипция успешно завершена. Длина текста: {len(transcript.strip())} символов")
            return transcript.strip()
        except Exception as e:
            logger.error(f"Ошибка при транскрипции: {str(e)}")
            return f"Ошибка при транскрипции: {str(e)}"
    
    def transcribe_audio_bytes(self, 
                              audio_bytes: bytes, 
                              language: str = "ru",
                              model: str = "gpt-4o-mini-transcribe") -> str:
        """
        Транскрипция аудио из байтов с использованием OpenAI API
        
        Args:
            audio_bytes: Аудио данные в байтах
            language: Язык для транскрипции (код языка, например "ru" для русского)
            model: Модель для транскрипции
            
        Returns:
            Транскрибированный текст
        """
        try:
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # Транскрибируем
            result = self.transcribe_audio(temp_file_path, language, model)
            
            # Удаляем временный файл
            os.unlink(temp_file_path)
            
            return result
        except Exception as e:
            return f"Ошибка при транскрипции аудио: {str(e)}"