import os
import time
import traceback
import uuid
import warnings

from aiohttp import web

from model.model_app.handler.standard_model import StandardModel
from model.model_app.handler.decorator import request_handler
from src import logger
from src.assistant import Assistant

assistant = Assistant(
    service_cfg_path=os.environ.get("MODEL_CONFIG_PATH", "configs/service_configs/deployment_config.yaml"),
    swagger_path=os.environ.get("SWAGGER_PATH", "api/api.yaml")
)

warnings.filterwarnings("ignore", message=".*Unverified HTTPS request is being made.*")


class AssistantModel(StandardModel):
    """
    Assistant model implementation.

    This is an implementation of the StandardModel interface.

    So it implements:
        * predict() method with the initial_data as a parameter;
    """

    @request_handler
    def predict(self, params, **context):
        """
        Make prediction.

        :param initial_data: simple model initial data.
        :param context: request context.
        :return:
        Prediction result.
        In this simple case it is just a byte array representation of a
        constant value = 'Prediction result'.
        """
        call_id = uuid.uuid4()
        
        # logger.debug(f"REQUEST: CallId {call_id}: {json.dumps(anonymize_logs(params), ensure_ascii=False)}")
        status_code = 200

        data = params.get("structured_data", {})
        configuration = params.get("configuration", {})

        if assistant is None:
            result = {
                "error": str(web.HTTPException(reason="Assistant model is unavailable")),
                "stackTrace": None
            }
            logger.exception("Service error: " + str(result))
            model_execution_time = 0
            status_code = 503
            return result, model_execution_time, status_code

        logger.info(f'Start processing request; CallId {call_id}')
        shutdown_from_error = False
        try:
            since = time.time()
            logger.info(f'Getting result from model; CallId {call_id}')
            if data["request_type"] == "single_eval":
                result, meta_result = assistant.predict_single_eval(data["request"], configuration)
            elif data["request_type"] == "batch_eval":
                result, meta_result = assistant.predict_batch_eval(data, configuration)
            else:
                msg = f"Unknown request_type: {data['request_type']}. Should be 'single_eval' or 'batch_eval'."
                raise ValueError(msg)
            result = {
                "response_type": data["request_type"],
                "response": result,
                "skill_meta_result": meta_result
            }
            model_execution_time = time.time() - since
        except TimeoutError as ex:
            result = {
                "error": str(ex),
                "stackTrace": traceback.format_exc()
            }
            logger.exception(f"CallId {call_id} Assistant model error: " + str(result))
            model_execution_time = time.time() - since
            status_code = 408
        except BrokenPipeError as ex:
            result = {
                "error": 'Possible OOM, setting shutdown. Error:' + str(ex),
                "stackTrace": traceback.format_exc()
            }
            logger.exception(f"Possible OOM, setting shutdown. CallId {call_id}")
            logger.exception("Service error: " + str(result))
            model_execution_time = 0
            status_code = 500
            shutdown_from_error = True
        except Exception as ex:
            result = {
                "error": str(ex),
                "stackTrace": traceback.format_exc()
            }
            logger.exception(f"CallId {call_id} Assistant model error: " + str(result))
            model_execution_time = 0
            status_code = 400

        shutdown_event = context['shutdown_event']

        if shutdown_from_error:
            shutdown_event.set()

        if shutdown_event.is_set():
            logger.info('Cancelled')
            return result, model_execution_time, status_code

        return result, model_execution_time, status_code

    @request_handler
    def version(self, *args, **kwargs):
        with open('VERSION') as f:
            version: str = f.read().strip()
        return {
            'version': version,
        }


if __name__ == "__main__":
    from model.model_app import app
    from model_assistant import assistant_model_config

    app.run(model=AssistantModel(), configs={assistant_model_config})
