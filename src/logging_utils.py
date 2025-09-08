from copy import deepcopy
import os
import re
from typing import Any


def anonymize_logs(d: Any) -> Any:
    if os.environ.get("LOG_DEBUG_HAZE", "false").lower() == "false":
        return d
    if not isinstance(d, str):
        d_copy = deepcopy(d)
    else:
        d_copy = d
    return recursive_anonymize(d_copy)


def recursive_anonymize(d: Any) -> Any:
    if isinstance(d, dict):
        for k in d:
            d[k] = recursive_anonymize(d[k])
        return d
    if isinstance(d, list) or isinstance(d, tuple):
        return [recursive_anonymize(el) for el in d]
    elif isinstance(d, str):
        return re.sub(r"[\w\d]", "?", d)
    elif isinstance(d, float) or isinstance(d, int):
        return "?"
    return d
