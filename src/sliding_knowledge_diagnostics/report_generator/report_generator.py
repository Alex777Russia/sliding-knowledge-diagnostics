import json
import os
import math
from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
from typing import Dict, List

from src.sliding_knowledge_diagnostics.report_generator.prompts import (
    REPORT_GENERATOR_SYSTEM_PROMPT, 
    REPORT_GENERATOR_USER_TEMPLATE
)
from src.sliding_knowledge_diagnostics.utils import HistoryElement
from src.llm import LLM
from src.logging_config import get_logger
from src.audio import VoiceAnalyzer

from markdown_pdf import MarkdownPdf, Section

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(
            self
    ):
        logger.info("Инициализация ReportGenerator")
        self.llm = LLM()
        self.system_prompt = REPORT_GENERATOR_SYSTEM_PROMPT
        self.user_template = REPORT_GENERATOR_USER_TEMPLATE
        
        # Инициализация анализатора голоса для отчета
        try:
            self.voice_analyzer = VoiceAnalyzer()
            logger.info("VoiceAnalyzer для отчета успешно инициализирован")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать VoiceAnalyzer: {e}")
            self.voice_analyzer = None
        
        logger.info("ReportGenerator успешно инициализирован")

    def generate_report(
            self,
            history: List[Dict[str, str]]
    ) -> str:
        logger.info("Генерация отчета по результатам экзамена")
        logger.debug(f"Количество элементов в истории: {len(history)}")
        
        structured_data = self._extract_data_for_report(
            history=history
        )
        logger.debug(f"Структурированные данные подготовлены. Размер: {len(structured_data)} символов")
        
        messages = [
            {
                "role": "system",
                "text": REPORT_GENERATOR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "text": REPORT_GENERATOR_USER_TEMPLATE.format(
                    structured_data=structured_data
                )
            }
        ]

        logger.info("Отправляем запрос к LLM для генерации отчета")
        report = self.llm.run(
            messages=messages
        )
        logger.info(f"Отчет сгенерирован. Длина отчета: {len(report)} символов")
        logger.debug(f"Отчет: {report[:200]}...")

        logger.info(f"Формируем pdf файл с отчетом")
        pdf_path = self._get_pdf_report(report, history)
        logger.info(f"pdf файл с отчетом сформирован")
        logger.debug(f"Путь к pdf файлу с отчетом: {pdf_path}")

        return report
    
    def _get_pdf_report(self, report: str, history: List[HistoryElement]) -> str:
        """
        Генерация PDF-файла с текстовым отчетом и радар-диаграммой по уровням Блума.
        Возвращает путь к созданному PDF.
        """
        
        level_scores = defaultdict(list)
        for h in history:
            lvl = getattr(h, "blum_level", None)
            score = getattr(h, "evaluation_result", None).evaluation_score if getattr(h, "evaluation_result", None) else None
            if lvl and score is not None:
                level_scores[lvl].append(float(score))
                logger.info(f'{level_scores[lvl]}')

        bloom_levels = ["Знание", "Понимание", "Применение", "Анализ", "Синтез", "Оценка"]
        avg_scores = {}
        for lvl in bloom_levels:
            vals = level_scores.get(lvl, [])
            avg_scores[lvl] = sum(vals) / len(vals) if vals else 0.0

        chart_path = self._generate_radar_chart(bloom_levels, avg_scores)
        pdf_path = self._generate_pdf(report, chart_path)

        return pdf_path

    def _generate_radar_chart(self, bloom_levels: List[str], avg_scores: Dict[str, float]) -> str:
        labels = bloom_levels
        values = [avg_scores[l] for l in labels]
        values += values[:1]
        angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
        plt.ylim(0, 1)

        ax.plot(angles, values, linewidth=2, linestyle="solid")
        ax.fill(angles, values, alpha=0.25)

        chart_path = os.path.join("bloom_radar.png")
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close(fig)

        return chart_path

    def _generate_pdf(self, report_md: str, chart_path: str, output_pdf: str = "report.pdf"):
        """
        Генерирует PDF из Markdown и добавляет график как изображение 
        в отдельной секции, используя markdown-pdf.
        
        :param report_md: Отчёт в Markdown формате (строка).
        :param chart_path: Путь к картинке графика (png/jpg/pdf).
        :param output_pdf: Имя результирующего PDF-файла.
        """
        # Создаем объект PDF
        pdf = MarkdownPdf(
            toc_level=2,
            optimize=True
        )
        
        # Метаданные (опционально)
        pdf.meta["title"] = "Отчет по результатам экзамена"
        pdf.meta["author"] = "Студент / Преподаватель"
        
        pdf.add_section(Section(report_md, toc=True))
        
        # Добавляем секцию с графиком
        # Чтобы вставить изображение, Markdown поддерживает синтаксис ![alt](path)
        chart_md = f"## Уровни Блума — Radar Chart\n\n![chart]({chart_path})\n"
        pdf.add_section(Section(chart_md, toc=False))
        
        # Сохраняем PDF
        pdf.save(output_pdf)
        return output_pdf


    def _extract_data_for_report(
        self,
        history: List[HistoryElement]
    ) -> str:
        all_data = []
        for i in range(0, len(history) - 1, 2):
            question_element, answer_element = history[i], history[i + 1]
            
            # Проверяем, был ли это голосовой ответ
            voice_analysis = ""
            if hasattr(answer_element, 'audio_file_path') and answer_element.audio_file_path:
                voice_analysis = self._analyze_voice_for_report(answer_element.audio_file_path)
            
            data_item = {
                "Вопрос": question_element.content,
                "Ответ студента": answer_element.content,
                "Верный ответ": question_element.gt_answer,
                "Уровень таксономии Блума": question_element.blum_level,
                "Оценка ответа студента": answer_element.evaluation_result.evaluation_score,
                "Комментарий по поводу оценки": answer_element.evaluation_result.evaluation_comment
            }
            
            # Добавляем анализ голоса с учетом комментария оценщика, если он был выполнен
            if voice_analysis:
                enhanced_voice_analysis = self._enhance_voice_analysis_with_evaluation(
                    voice_analysis, 
                    answer_element.evaluation_result.evaluation_comment,
                    answer_element.evaluation_result.evaluation_score
                )
                data_item["Анализ характеристик голоса"] = enhanced_voice_analysis
            
            all_data.append(data_item)

        return json.dumps(all_data, indent=4, ensure_ascii=False)
    
    def _analyze_voice_for_report(self, audio_file_path: str) -> str:
        """
        Анализирует голос для включения в отчет
        """
        if not self.voice_analyzer or not os.path.exists(audio_file_path):
            return "Анализ голоса недоступен"
        
        try:
            logger.info(f"Анализируем голос для отчета: {audio_file_path}")
            voice_features = self.voice_analyzer.analyze_audio(audio_file_path)
            return self.voice_analyzer.format_voice_analysis_for_llm(voice_features)
        except Exception as e:
            logger.error(f"Ошибка при анализе голоса для отчета: {e}")
            return "Ошибка анализа голоса"
    
    def _enhance_voice_analysis_with_evaluation(self, voice_analysis: str, evaluation_comment: str, evaluation_score: float) -> str:
        """
        Улучшает анализ голоса с учетом комментария и оценки от LLM
        """
        try:
            # Создаем контекст для более точной интерпретации
            context = f"""
                Контекст для интерпретации голосовых характеристик:
                - Оценка ответа: {evaluation_score:.2f}/1.0
                - Комментарий оценщика: {evaluation_comment}

                Исходный анализ голоса:
                {voice_analysis}

                Проанализируй голосовые характеристики с учетом оценки и комментария оценщика. 
                Учти, что:
                - Высокая оценка + проблемы с голосом = возможная заученность
                - Низкая оценка + уверенный голос = поверхностное знание
                - Средняя оценка + неуверенный голос = поиск знаний
                - Высокая оценка + уверенный голос = глубокое понимание
            """
                            
            # Используем LLM для улучшенной интерпретации
            messages = [
                {
                    "role": "system",
                    "text": """Ты - эксперт по анализу голосовых характеристик в образовательном контексте. 
                    Твоя задача - интерпретировать технические характеристики голоса с учетом оценки ответа студента.
                    
                    Учитывай:
                    - Оценку ответа (0-1) и комментарий оценщика
                    - Технические характеристики голоса (темп, паузы, стабильность, интонация, эмоции)
                    - Связь между голосовыми характеристиками и качеством знаний
                    
                    Дай краткий, но содержательный анализ (2-3 предложения) с конкретными выводами."""
                },
                {
                    "role": "user", 
                    "text": context
                }
            ]
            
            enhanced_analysis = self.llm.run(messages)
            return f"🎚 Анализ характеристик голоса:\n\n{enhanced_analysis}"
            
        except Exception as e:
            logger.error(f"Ошибка при улучшении анализа голоса: {e}")
            return voice_analysis  # Возвращаем исходный анализ при ошибке
