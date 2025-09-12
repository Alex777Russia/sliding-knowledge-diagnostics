import librosa
import numpy as np
from typing import Dict
from src.logging_config import get_logger

logger = get_logger(__name__)


class VoiceAnalyzer:
    def analyze_audio(self, audio_file_path: str) -> Dict:
        try:
            y, sr = librosa.load(audio_file_path, sr=None)
            duration = len(y) / sr
            
            features = {
                'duration': duration,
                'sample_rate': sr,
                'speech_rate': self._calculate_speech_rate(y, sr),
                'pauses': self._analyze_pauses(y, sr),
                'phonetics': self._analyze_phonetics(y, sr),
                'prosody': self._analyze_prosody(y, sr),
                'emotions': self._analyze_emotions(y, sr)
            }
            
            return features
            
        except Exception as e:
            return self._get_default_features()
    
    def _calculate_speech_rate(self, y: np.ndarray, sr: int) -> Dict:
        try:
            intervals = librosa.effects.split(y, top_db=20)
            
            if len(intervals) == 0:
                return {'words_per_minute': 0, 'speech_segments': 0}
            
            total_speech_time = sum((end - start) / sr for start, end in intervals)
            estimated_words = int(total_speech_time * 2.5)  # 2.5 слова в секунду
            words_per_minute = (estimated_words / total_speech_time * 60) if total_speech_time > 0 else 0
            
            return {
                'words_per_minute': round(words_per_minute, 1),
                'speech_segments': len(intervals),
                'total_speech_time': round(total_speech_time, 2),
                'speech_ratio': round(total_speech_time / (len(y) / sr), 2)
            }
            
        except Exception as e:
            return {'words_per_minute': 0, 'speech_segments': 0}
    
    def _analyze_pauses(self, y: np.ndarray, sr: int) -> Dict:
        try:
            intervals = librosa.effects.split(y, top_db=20)
            
            if len(intervals) <= 1:
                return {'long_pauses_count': 0, 'avg_pause_length': 0, 'pause_ratio': 0}
            
            pauses = []
            for i in range(len(intervals) - 1):
                pause_start = intervals[i][1]
                pause_end = intervals[i + 1][0]
                pause_length = (pause_end - pause_start) / sr
                pauses.append(pause_length)
            
            long_pauses = [p for p in pauses if p > 0.5]  
            avg_pause_length = np.mean(pauses) if pauses else 0
            total_pause_time = sum(pauses)
            total_time = len(y) / sr
            pause_ratio = total_pause_time / total_time if total_time > 0 else 0
            
            return {
                'long_pauses_count': len(long_pauses),
                'avg_pause_length': round(avg_pause_length, 2),
                'pause_ratio': round(pause_ratio, 2),
                'total_pauses': len(pauses)
            }
            
        except Exception as e:
            return {'long_pauses_count': 0, 'avg_pause_length': 0, 'pause_ratio': 0}
    
    def _analyze_phonetics(self, y: np.ndarray, sr: int) -> Dict:
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            mfcc_std = np.std(mfccs, axis=1)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            mfcc_variation = np.mean(mfcc_std)
            centroid_variation = np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-8)
            rolloff_variation = np.std(spectral_rolloff) / (np.mean(spectral_rolloff) + 1e-8)
            
            speech_stability = 1 - min(1, (mfcc_variation + centroid_variation + rolloff_variation) / 3)
            
            return {
                'speech_stability': round(speech_stability, 2),
                'spectral_variation': round(mfcc_variation, 2),
                'centroid_variation': round(centroid_variation, 2),
                'rolloff_variation': round(rolloff_variation, 2)
            }
            
        except Exception as e:
            return {'speech_stability': 0, 'spectral_variation': 0}
    
    def _analyze_prosody(self, y: np.ndarray, sr: int) -> Dict:
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return {'pitch_variation': 0, 'energy_variation': 0, 'monotony': 1}
            
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_variation = pitch_std / pitch_mean if pitch_mean > 0 else 0
            
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            energy_variation = energy_std / energy_mean if energy_mean > 0 else 0
            
            monotony = 1 - min(1, (pitch_variation + energy_variation) / 2)
            
            return {
                'pitch_variation': round(pitch_variation, 2),
                'energy_variation': round(energy_variation, 2),
                'monotony': round(monotony, 2),
                'avg_pitch': round(pitch_mean, 1),
                'avg_energy': round(energy_mean, 3)
            }
            
        except Exception as e:
            return {'pitch_variation': 0, 'energy_variation': 0, 'monotony': 1}
    
    def _analyze_emotions(self, y: np.ndarray, sr: int) -> Dict:
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            avg_centroid = np.mean(spectral_centroid)
            avg_zcr = np.mean(zero_crossing_rate)
            mfcc_energy = np.mean(mfccs[0])  
            
            emotions = {
                'excitement': 0,  
                'calmness': 0,    
                'tension': 0,     
                'confidence': 0  
            }
            
            if avg_centroid > np.percentile(spectral_centroid, 75):
                emotions['excitement'] += 0.3
            else:
                emotions['calmness'] += 0.3
                
            if avg_zcr > np.percentile(zero_crossing_rate, 75):
                emotions['tension'] += 0.3
            else:
                emotions['confidence'] += 0.3
                
            if mfcc_energy > np.percentile(mfccs[0], 75):
                emotions['excitement'] += 0.2
                emotions['confidence'] += 0.2
            else:
                emotions['calmness'] += 0.2
                emotions['tension'] += 0.2
            
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: round(v / total, 2) for k, v in emotions.items()}
            
            dominant_emotion = max(emotions, key=emotions.get)
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'confidence': emotions[dominant_emotion]
            }
            
        except Exception as e:
            return {
                'emotions': {'excitement': 0.25, 'calmness': 0.25, 'tension': 0.25, 'confidence': 0.25},
                'dominant_emotion': 'neutral',
                'confidence': 0.25
            }
    
    def _get_default_features(self) -> Dict:
        return {
            'duration': 0,
            'sample_rate': 0,
            'speech_rate': {'words_per_minute': 0, 'speech_segments': 0},
            'pauses': {'long_pauses_count': 0, 'avg_pause_length': 0, 'pause_ratio': 0},
            'phonetics': {'speech_stability': 0, 'spectral_variation': 0},
            'prosody': {'pitch_variation': 0, 'energy_variation': 0, 'monotony': 1},
            'emotions': {
                'emotions': {'excitement': 0.25, 'calmness': 0.25, 'tension': 0.25, 'confidence': 0.25},
                'dominant_emotion': 'neutral',
                'confidence': 0.25
            }
        }
    
    def format_voice_analysis_for_llm(self, voice_features: Dict) -> str:
        try:
            analysis_text = "🎚 Анализ характеристик голоса:\n\n"
            
            speech_rate = voice_features.get('speech_rate', {})
            wpm = speech_rate.get('words_per_minute', 0)
            analysis_text += f"1. Темп речи: {wpm} слов/мин\n"
            if wpm > 200:
                analysis_text += "   → Слишком быстро (возможна поверхностность или волнение)\n"
            elif wpm < 100:
                analysis_text += "   → Слишком медленно (возможна неуверенность или обдумывание)\n"
            else:
                analysis_text += "   → Нормальный темп речи\n"
            
            pauses = voice_features.get('pauses', {})
            long_pauses = pauses.get('long_pauses_count', 0)
            avg_pause = pauses.get('avg_pause_length', 0)
            analysis_text += f"\n2. Паузы: {long_pauses} длинных пауз (>0.5с), средняя длина: {avg_pause}с\n"
            if long_pauses > 3:
                analysis_text += "   → Много длинных пауз (возможен поиск слов, слабое знание)\n"
            elif long_pauses == 0:
                analysis_text += "   → Мало пауз (возможна заученность или спешка)\n"
            else:
                analysis_text += "   → Нормальное количество пауз\n"
            
            phonetics = voice_features.get('phonetics', {})
            stability = phonetics.get('speech_stability', 0)
            analysis_text += f"\n3. Фонетика: стабильность речи {stability:.2f}\n"
            if stability < 0.3:
                analysis_text += "   → Низкая стабильность (возможны запинки, неуверенность)\n"
            elif stability > 0.7:
                analysis_text += "   → Высокая стабильность (уверенная речь)\n"
            else:
                analysis_text += "   → Средняя стабильность речи\n"
            
            prosody = voice_features.get('prosody', {})
            pitch_var = prosody.get('pitch_variation', 0)
            monotony = prosody.get('monotony', 1)
            analysis_text += f"\n4. Интонация: вариативность pitch {pitch_var:.2f}, монотонность {monotony:.2f}\n"
            if monotony > 0.7:
                analysis_text += "   → Высокая монотонность (низкая вовлеченность или заучивание)\n"
            elif monotony < 0.3:
                analysis_text += "   → Выраженные интонации (уверенность, понимание)\n"
            else:
                analysis_text += "   → Нормальная интонационная вариативность\n"
            
            emotions = voice_features.get('emotions', {})
            dominant = emotions.get('dominant_emotion', 'neutral')
            confidence = emotions.get('confidence', 0)
            analysis_text += f"\n5. Эмоции: доминирующая эмоция - {dominant} (уверенность: {confidence:.2f})\n"
            
            emotion_map = {
                'excitement': 'волнение/энтузиазм',
                'calmness': 'спокойствие',
                'tension': 'напряжение/стресс',
                'confidence': 'уверенность',
                'neutral': 'нейтральное состояние'
            }
            analysis_text += f"   → {emotion_map.get(dominant, dominant)}\n"
            
            return analysis_text
            
        except Exception as e:
            return "🎚 Анализ характеристик голоса: недоступен"
