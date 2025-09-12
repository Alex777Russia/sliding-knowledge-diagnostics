import os

from typing import List
from yandex_cloud_ml_sdk import YCloudML

from src.custom_logger import log_function_call
from src.llm.utils import LLMMessage


class LLM:
    def __init__(
            self
    ):
        self.folder_id = os.getenv("FOLDER_ID", None)
        self.api_key = os.getenv("YANDEX_API_KEY", None)
        self.model = self._create_sdk()
    
    @log_function_call
    def _create_sdk(
            self
    ) -> YCloudML:
        n_resps = 3
        while n_resps > 0:
            try:
                sdk = YCloudML(
                    folder_id=self.folder_id,
                    auth=self.api_key
                )
                return sdk
            except Exception as e:
                last_e = e
                n_resps -= 1

        raise RuntimeError(f'LLM Error - {last_e}')
    
    @log_function_call(log_result=True)
    def run(
            self,
            messages: List[LLMMessage],
            model_name: str = "yandexgpt"
    ) -> str:        
        try:
            result = (
                self.model.models.completions(model_name).configure(temperature=0.5).run(messages)
            )
            response_text = result[0].text
            return response_text
        except Exception as e:
            raise
