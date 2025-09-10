"""
Централизованная конфигурация логгирования для проекта Sliding Knowledge Diagnostics
"""
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Настройка централизованного логгирования для всего приложения
    
    Args:
        log_level: Уровень логгирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Логировать ли в файл
        log_dir: Директория для логов
    
    Returns:
        Настроенный логгер
    """
    # Создаем директорию для логов если её нет
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
    
    # Получаем уровень логгирования
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Создаем форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Очищаем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Файловый обработчик
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"sliding_knowledge_diagnostics_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Также создаем общий лог файл (перезаписывается при каждом запуске)
        general_log_file = log_path / "app.log"
        general_file_handler = logging.FileHandler(general_log_file, mode='w', encoding='utf-8')
        general_file_handler.setLevel(numeric_level)
        general_file_handler.setFormatter(formatter)
        root_logger.addHandler(general_file_handler)
    
    # Настраиваем логгеры для внешних библиотек
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("yandex_cloud_ml_sdk").setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Логгирование настроено. Уровень: {log_level}")
    if log_to_file:
        logger.info(f"Логи сохраняются в директории: {log_path.absolute()}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Получить логгер для конкретного модуля
    
    Args:
        name: Имя модуля (обычно __name__)
    
    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)


# Инициализация логгирования при импорте модуля
if not logging.getLogger().handlers:
    setup_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_to_file=os.getenv("LOG_TO_FILE", "true").lower() == "true"
    )
