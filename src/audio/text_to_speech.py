import os
import tempfile
import logging
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Класс для генерации речи из текста с использованием OpenAI API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация TTS генератора
        
        Args:
            api_key: API ключ OpenAI (если не указан, будет взят из переменной окружения OPENAI_API_KEY)
        """
        logger.info("Инициализация OpenAI клиента для генерации речи")
        self.client = OpenAI(api_key=api_key)
        logger.info("OpenAI клиент для TTS успешно инициализирован")
    
    def generate_speech(self, 
                       text: str, 
                       voice: str = "alloy",
                       model: str = "tts-1",
                       output_format: str = "mp3") -> Optional[str]:
        """
        Генерация речи из текста с использованием OpenAI API
        
        Args:
            text: Текст для озвучивания
            voice: Голос для озвучивания (alloy, echo, fable, onyx, nova, shimmer)
            model: Модель TTS (tts-1, tts-1-hd)
            output_format: Формат выходного файла (mp3, opus, aac, flac)
            
        Returns:
            Путь к сгенерированному аудио файлу или None в случае ошибки
        """
        logger.info(f"Начинаем генерацию речи для текста длиной {len(text)} символов")
        
        if not text or not isinstance(text, str):
            logger.error("Неверный текст для озвучивания")
            return None
        
        if len(text.strip()) == 0:
            logger.error("Пустой текст для озвучивания")
            return None
        
        # Ограничиваем длину текста (OpenAI TTS имеет ограничение в 4096 символов)
        if len(text) > 4096:
            logger.warning(f"Текст слишком длинный ({len(text)} символов), обрезаем до 4096")
            text = text[:4096]
        
        try:
            logger.info(f"Отправляем запрос в OpenAI TTS API (модель: {model}, голос: {voice})")
            
            # Генерируем речь
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=output_format
            )
            
            # Создаем временный файл для сохранения аудио
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            logger.info(f"Речь успешно сгенерирована и сохранена в: {temp_file_path}")
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Ошибка при генерации речи: {str(e)}")
            return None
    
    def generate_speech_for_question(self, question_text: str) -> Optional[str]:
        """
        Специальный метод для генерации речи для вопросов экзамена
        
        Args:
            question_text: Текст вопроса для озвучивания
            
        Returns:
            Путь к сгенерированному аудио файлу или None в случае ошибки
        """
        # Очищаем текст от эмодзи и специальных символов для лучшего озвучивания
        clean_text = self._clean_text_for_speech(question_text)
        
        # Используем более подходящий голос для образовательного контента
        return self.generate_speech(
            text=clean_text,
            voice="nova",  # Более четкий голос для образовательного контента
            model="tts-1-hd",  # Высокое качество
            output_format="mp3"
        )
    
    def _clean_text_for_speech(self, text: str) -> str:
        """
        Очистка текста для лучшего озвучивания
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        # Удаляем эмодзи микрофона и специальные символы
        text = text.replace("🎤", "")
        text = text.replace("**", "")  # Убираем жирный текст
        text = text.replace("*", "")   # Убираем курсив
        
        # Убираем лишние пробелы и переносы строк
        text = " ".join(text.split())
        
        return text.strip()
    
    def cleanup_audio_file(self, file_path: str) -> None:
        """
        Удаление временного аудио файла
        
        Args:
            file_path: Путь к файлу для удаления
        """
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Временный аудио файл удален: {file_path}")
        except Exception as e:
            logger.warning(f"Не удалось удалить временный файл {file_path}: {str(e)}")
