import logging
import sys
import functools
import time
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)

PARAM_MAX_LEN = 70

def log_function_call(_func=None, *, log_result=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            call_id = uuid.uuid4()

            args_repr = []
            if args and hasattr(args[0], func.__name__):
                args_repr.append(f"self=<{args[0].__class__.__name__}>")
                args_repr.extend([cut_param(repr(a)) for a in args[1:]])
            else:
                args_repr = [cut_param(repr(a)) for a in args]

            kwargs_repr = [f"{k}={cut_param(repr(v))}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            logging.info(f"CallId: {call_id} | Calling function: {func.__name__}({signature})")
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                if log_result:
                    logging.info(
                        f"CallId: {call_id} | Got output from function: {func.__name__} | "
                        f"Function output: {result!r} | Execution time: {execution_time:.4f} с"
                    )
                else:
                    logging.info(
                        f"CallId: {call_id} | Got output from function: {func.__name__} | "
                        f"Execution time: {execution_time:.4f} с"
                    )
                return result
            except Exception as e:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                logging.error(
                    f"CallId: {call_id} | Error in function: {func.__name__} | "
                    f"Exeception: {e!r} | Execution time before error: {execution_time:.4f} с"
                )
                raise e

        return wrapper
    
    if _func is None:
        return decorator
    else:
        return decorator(_func)


def cut_param(param) -> str:
    if len(param) > PARAM_MAX_LEN:
        return f"{param[:PARAM_MAX_LEN]}..."
    return param
