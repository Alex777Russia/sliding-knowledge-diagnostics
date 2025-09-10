import logging
import os

from typing import List, Optional
from yandex_cloud_ml_sdk import YCloudML

from src.llm.utils import LLMMessage
from src.logging_config import get_logger

logger = get_logger(__name__)


class LLM:
    def __init__(
            self
    ):
        logger.info("Инициализация LLM клиента")
        self.folder_id = os.getenv("FOLDER_ID", None)
        self.api_key = os.getenv("YANDEX_API_KEY", None)
        
        if not self.folder_id:
            logger.error("FOLDER_ID не установлен в переменных окружения")
        if not self.api_key:
            logger.error("YANDEX_API_KEY не установлен в переменных окружения")
            
        self.model = self._create_sdk()
        logger.info("LLM клиент успешно инициализирован")
    
    def _create_sdk(
            self
    ) -> YCloudML:
        logger.info("Создание YCloudML SDK")
        n_resps = 3
        while n_resps > 0:
            try:
                logger.debug(f"Попытка создания SDK. Осталось попыток: {n_resps}")
                sdk = YCloudML(
                    folder_id=self.folder_id,
                    auth=self.api_key
                )
                logger.info("YCloudML SDK успешно создан")
                return sdk
            except Exception as e:
                logger.warning(f"Ошибка создания SDK: {e}")
                last_e = e
                n_resps -= 1

        logger.error(f"Не удалось создать YCloudML SDK после 3 попыток. Последняя ошибка: {last_e}")
        raise RuntimeError(f'LLM Error - {last_e}')
    
    def run(
            self,
            messages: List[LLMMessage],
            model_name: str = "yandexgpt"
    ) -> str:
        logger.info(f"Отправка запроса к модели {model_name}")
        
        try:
            result = (
                self.model.models.completions(model_name).configure(temperature=0.5).run(messages)
            )
            response_text = result[0].text
            logger.info(f"Ответ получен: {response_text[:200]}...")
            return response_text
        except Exception as e:
            logger.error(f"Ошибка при выполнении запроса к модели: {e}")
            raise
