from dataclasses import dataclass


@dataclass
class LLMMessage:
    role: str
    text: str
