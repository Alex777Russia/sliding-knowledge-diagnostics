import functools
import json
import os
import time
from types import TracebackType
from typing import Any, Callable, Optional, Type
import uuid

from src.logging_utils import anonymize_logs


class Timing:
    def __init__(self, process_name: Optional[str] = None, logger: Optional[Any] = None) -> None:
        self.logger = logger if logger else type("", (object,), {"info": print})
        self.process_name = "_".join(str(process_name).split())

    def __enter__(self) -> None:
        self.call_id = uuid.uuid4()
        msg = "CallId: {} Call: {}".format(self.call_id, self.process_name)
        self.logger.info(msg)
        self.start_time = time.time()

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        execution_time = time.time() - self.start_time
        msg = "CallId: {} Call: {} Elapsed: {:.4f}s".format(self.call_id, self.process_name, execution_time)
        self.logger.info(msg)

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrap(*args: Any, **kwargs: Any) -> Any:
            call_id = uuid.uuid4()
            self.logger.info(self._get_function_start_message(func, call_id))
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            self.logger.info(self._get_function_end_message(func, execution_time, call_id))
            if result is not None:
                short_info = self.get_short_output_stats(result, func, call_id)
                self.logger.info(short_info)

            return result

        return wrap

    def _get_function_start_message(self, func: Callable, call_id: uuid.UUID) -> str:
        msg = "CallId: {} Call: {}".format(call_id, func.__qualname__)
        return msg

    def _get_function_end_message(self, func: Callable, execution_time: float, call_id: uuid.UUID) -> str:
        msg = "CallId: {} Call: {} Elapsed: {:.4f}s".format(call_id, func.__qualname__, execution_time)
        return msg

    @staticmethod
    def get_atts_dict_info(d: dict) -> str:
        attr_info = ""
        for key, value in d.items():
            attr_info += f" {key}: {value.__class__.__name__}"
            if hasattr(value, "__len__"):
                attr_info += f" (len={len(value)});"
            else:
                attr_info += ";"
        return attr_info.strip()

    def get_short_output_stats(self, result: Any, func: Callable, call_id: uuid.UUID):
        debug_type = (
            result.__class__.__name__.capitalize()
            if not result.__class__.__name__[0].isupper()
            else result.__class__.__name__
        )
        short_info = f"CallId: {call_id} Call: {func.__qualname__} Output type: {debug_type};"
        if result is None:
            return short_info
        if result.__class__.__name__[0].isupper():
            short_info += f"\nAttributes: {self.get_atts_dict_info(result.__dict__)}"
        if isinstance(result, list):
            if all(el is not None for el in result):
                short_info += f" Len = {len(result)}; Item type: {result[0].__class__.__name__ if result else None}"
                if result and result[0].__class__.__name__[0].isupper():
                    short_info += f"\nItem 0 attributes: {self.get_atts_dict_info(result[0].__dict__)}"
        if isinstance(result, tuple):
            short_info += f" Len = {len(result)}; Items:"
            for i, el in enumerate(result):
                if el is None:
                    continue
                short_info += f"\nItem {i} type: {el.__class__.__name__ if result else None}; "
                if result and el.__class__.__name__[0].isupper():
                    short_info += f"Attributes: {self.get_atts_dict_info(el.__dict__)}"
        elif isinstance(result, dict):
            short_info += f"\nKey-Values: {self.get_atts_dict_info(result)}"
        return short_info
