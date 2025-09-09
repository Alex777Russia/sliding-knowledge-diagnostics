import logging
import os

from typing import List, Optional
from yandex_cloud_ml_sdk import YCloudML

from src.llm.utils import LLMMessage


class LLM:
    def __init__(
            self
    ):
        self.folder_id = os.getenv("FOLDER_ID", None)
        self.api_key = os.getenv("YANDEX_API_KEY", None)
        self.model = self._create_sdk()
    
    def _create_sdk(
            self
    ) -> YCloudML:
        n_resps = 3
        while n_resps > 0:
            try:
                return YCloudML(
                    folder_id=self.folder_id,
                    auth=self.api_key
                )
            except Exception as e:
                logging.warning(e)
                last_e = e

        raise RuntimeError(f'LLM Error - {last_e}')
    
    def run(
            self,
            messages: List[LLMMessage],
            model_name: str = "yandexgpt"
    ) -> str:
        result = (
            self.model.models.completions(model_name).configure(temperature=0.5).run(messages)
        )

        return result[0].text
