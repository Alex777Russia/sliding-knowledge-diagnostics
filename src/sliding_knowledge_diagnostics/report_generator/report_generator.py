import json
import os
import math
from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
from typing import Dict, List

from src.sliding_knowledge_diagnostics.answer_evaluator import AnswerEvaluator
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
        self.llm = LLM()
        self.answer_evaluator = AnswerEvaluator()

        try:
            self.voice_analyzer = VoiceAnalyzer()
        except Exception as e:
            self.voice_analyzer = None
        
    def generate_report(
            self,
            history: List[Dict[str, str]]
    ) -> str:
        structured_data = self._extract_data_for_report(
            history=history
        )
        
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

        report = self.llm.run(
            messages=messages
        )

        self._generate_pdf_report(report, history)

        return report
    
    def _generate_pdf_report(
            self, 
            report: str, 
            history: List[HistoryElement]
    ) -> None:
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
        self._generate_pdf(report, chart_path)

    def _generate_radar_chart(
            self, 
            bloom_levels: List[str], 
            avg_scores: Dict[str, float]
    ) -> str:
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

    def _generate_pdf(
            self, 
            report_md: str, 
            chart_path: str, 
            output_pdf: str = "report.pdf"
    ) -> None:
        pdf = MarkdownPdf(
            toc_level=2,
            optimize=True
        )
        
        pdf.meta["title"] = "Отчет по результатам экзамена"
        pdf.meta["author"] = "Студент / Преподаватель"
        
        pdf.add_section(Section(report_md, toc=True))

        chart_md = f"## Уровни Блума — Radar Chart\n\n![chart]({chart_path})\n"
        pdf.add_section(Section(chart_md, toc=False))
        
        pdf.save(output_pdf)

    def _extract_data_for_report(
        self,
        history: List[HistoryElement]
    ) -> str:
        all_data = []
        for i in range(0, len(history) - 1, 2):
            question_element, answer_element = history[i], history[i + 1]
            
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
            
            if voice_analysis:
                enhanced_voice_analysis = self.answer_evaluator._enhance_voice_analysis_with_evaluation(
                    voice_analysis, 
                    answer_element.evaluation_result.evaluation_comment,
                    answer_element.evaluation_result.evaluation_score
                )
                data_item["Анализ характеристик голоса"] = enhanced_voice_analysis
            
            all_data.append(data_item)

        return json.dumps(all_data, indent=4, ensure_ascii=False)
    
    def _analyze_voice_for_report(self, audio_file_path: str) -> str:
        if not self.voice_analyzer or not os.path.exists(audio_file_path):
            return "Анализ голоса недоступен"
        
        try:
            voice_features = self.voice_analyzer.analyze_audio(audio_file_path)
            return self.voice_analyzer.format_voice_analysis_for_llm(voice_features)
        except Exception as e:
            return "Ошибка анализа голоса"
