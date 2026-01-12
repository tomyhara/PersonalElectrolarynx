#!/usr/bin/env python3
"""
人工咽頭チューニングシステム
ESP32との連携により音声パラメータを最適化
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
import json
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.fft import fft, fftfreq
import librosa
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import argparse
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """システム全体の設定"""
    # 音声処理設定
    sample_rate: int = 22050
    frame_size: int = 2048
    recording_duration: float = 1.5
    stabilization_delay: float = 0.5

    # 周波数解析設定
    f0_min: float = 80.0
    f0_max: float = 400.0
    f0_default: float = 150.0
    harmonic_count: int = 5
    harmonic_search_range: float = 10.0
    formant_count: int = 3
    formant_defaults: List[float] = field(default_factory=lambda: [700.0, 1200.0, 2500.0])

    # スペクトル解析設定
    savgol_window: int = 51
    savgol_order: int = 3
    peak_height_ratio: float = 0.1
    peak_distance: int = 50

    # 評価スコアの重み
    score_weight_spectrum: float = 0.4
    score_weight_harmonics: float = 0.4
    score_weight_formants: float = 0.2

    # PSO設定
    pso_n_particles: int = 8
    pso_max_iterations: int = 20
    pso_inertia: float = 0.5
    pso_c1: float = 1.5
    pso_c2: float = 1.5

    # パラメータ範囲
    rate_min: float = 0.5
    rate_max: float = 2.0
    rate_init_min: float = 0.8
    rate_init_max: float = 1.2
    amplitude_min: float = 0.1
    amplitude_max: float = 1.0
    amplitude_init_min: float = 0.5
    amplitude_init_max: float = 1.0
    harmonic_weight_min: float = 0.01
    harmonic_weight_max: float = 1.0
    harmonic_weight_init_min: float = 0.1
    harmonic_weight_init_max: float = 1.0

    # 通信設定
    esp32_timeout: float = 5.0

    # デフォルト倍音構成
    default_harmonic_weights: List[float] = field(default_factory=lambda: [1.0, 0.4, 0.2, 0.1, 0.05])

@dataclass
class AudioParameters:
    """音声パラメータクラス"""
    playback_rate: float = 1.0
    amplitude: float = 0.8
    formant_f1: float = 700.0  # 第1フォルマント
    formant_f2: float = 1200.0  # 第2フォルマント
    harmonic_weights: Optional[List[float]] = None
    config: Optional[SystemConfig] = None

    def __post_init__(self):
        if self.harmonic_weights is None:
            if self.config:
                self.harmonic_weights = self.config.default_harmonic_weights.copy()
            else:
                self.harmonic_weights = [1.0, 0.4, 0.2, 0.1, 0.05]

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'playback_rate': self.playback_rate,
            'amplitude': self.amplitude,
            'formant_f1': self.formant_f1,
            'formant_f2': self.formant_f2,
            'harmonic_weights': self.harmonic_weights
        }

class AudioAnalyzer:
    """音声解析クラス"""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.sample_rate = self.config.sample_rate
        self.frame_size = self.config.frame_size
        self.audio_queue: queue.Queue = queue.Queue()
        self.recording = False
        
    def analyze_spectrum(self, audio_data: np.ndarray) -> dict:
        """スペクトル解析"""
        # FFT解析
        window = signal.hann(len(audio_data))
        windowed_data = audio_data * window
        spectrum = np.abs(fft(windowed_data))
        freqs = fftfreq(len(audio_data), 1/self.sample_rate)
        
        # 正の周波数のみ取得
        positive_freqs = freqs[:len(freqs)//2]
        positive_spectrum = spectrum[:len(spectrum)//2]
        
        # 基本周波数検出
        f0 = self._detect_fundamental_frequency(positive_freqs, positive_spectrum)
        
        # 倍音解析
        harmonics = self._analyze_harmonics(positive_freqs, positive_spectrum, f0)
        
        # フォルマント解析
        formants = self._detect_formants(positive_freqs, positive_spectrum)
        
        # スペクトル重心
        spectral_centroid = np.sum(positive_freqs * positive_spectrum) / np.sum(positive_spectrum)
        
        return {
            'f0': f0,
            'harmonics': harmonics,
            'formants': formants,
            'spectral_centroid': spectral_centroid,
            'spectrum': positive_spectrum,
            'freqs': positive_freqs
        }
    
    def _detect_fundamental_frequency(self, freqs: np.ndarray, spectrum: np.ndarray) -> float:
        """基本周波数検出（オートコリレーション法）"""
        # 設定された範囲で最大ピークを探す
        mask = (freqs >= self.config.f0_min) & (freqs <= self.config.f0_max)
        masked_spectrum = spectrum.copy()
        masked_spectrum[~mask] = 0

        peak_idx = np.argmax(masked_spectrum)
        return freqs[peak_idx] if peak_idx > 0 else self.config.f0_default
    
    def _analyze_harmonics(self, freqs: np.ndarray, spectrum: np.ndarray, f0: float) -> List[float]:
        """倍音解析"""
        harmonics = []
        for i in range(1, self.config.harmonic_count + 1):
            harmonic_freq = f0 * i
            # 設定範囲内で最大値を探す
            search_range = self.config.harmonic_search_range
            mask = (freqs >= harmonic_freq - search_range) & (freqs <= harmonic_freq + search_range)
            if np.any(mask):
                harmonic_amplitude = np.max(spectrum[mask])
                harmonics.append(harmonic_amplitude)
            else:
                harmonics.append(0.0)

        # 基本波で正規化
        if len(harmonics) > 0 and harmonics[0] > 0:
            harmonics = [h / harmonics[0] for h in harmonics]

        return harmonics
    
    def _detect_formants(self, freqs: np.ndarray, spectrum: np.ndarray) -> List[float]:
        """フォルマント検出"""
        # スペクトルエンベロープを求める
        envelope = signal.savgol_filter(
            spectrum,
            self.config.savgol_window,
            self.config.savgol_order
        )

        # ピーク検出
        peaks, _ = signal.find_peaks(
            envelope,
            height=np.max(envelope) * self.config.peak_height_ratio,
            distance=self.config.peak_distance
        )

        formant_freqs = []
        for peak in peaks[:self.config.formant_count]:
            formant_freqs.append(freqs[peak])

        # 足りない場合はデフォルト値で埋める
        while len(formant_freqs) < self.config.formant_count:
            formant_freqs.append(self.config.formant_defaults[len(formant_freqs)])

        return formant_freqs[:self.config.formant_count]
    
    def start_recording(self, duration: Optional[float] = None) -> np.ndarray:
        """録音開始"""
        if duration is None:
            duration = self.config.recording_duration

        def record_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Recording status: {status}")
            self.audio_queue.put(indata.copy())

        self.recording = True
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=record_callback,
                blocksize=self.frame_size
            ):
                logger.info(f"Recording for {duration} seconds...")
                time.sleep(duration)
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            self.recording = False
            return np.array([])

        self.recording = False

        # 録音データを結合
        audio_blocks = []
        while not self.audio_queue.empty():
            audio_blocks.append(self.audio_queue.get())

        if audio_blocks:
            return np.concatenate(audio_blocks, axis=0).flatten()
        else:
            logger.warning("No audio data recorded")
            return np.array([])

class ESP32Controller:
    """ESP32制御クラス"""

    def __init__(self, esp32_ip: str, config: Optional[SystemConfig] = None):
        self.esp32_ip = esp32_ip
        self.base_url = f"http://{esp32_ip}"
        self.config = config or SystemConfig()

    def send_parameters(self, params: AudioParameters) -> bool:
        """パラメータをESP32に送信"""
        try:
            data = {
                "rate": params.playback_rate,
                "amplitude": params.amplitude,
                "harmonics": params.harmonic_weights
            }

            response = requests.post(
                f"{self.base_url}/update",
                json=data,
                timeout=self.config.esp32_timeout
            )

            if response.status_code == 200:
                logger.info("Parameters sent successfully")
                return True
            else:
                logger.error(f"Failed to send parameters: {response.status_code}")
                return False

        except requests.RequestException as e:
            logger.error(f"ESP32通信エラー: {e}")
            return False

    def start_playback(self) -> bool:
        """再生開始"""
        try:
            response = requests.post(
                f"{self.base_url}/play",
                timeout=self.config.esp32_timeout
            )
            if response.status_code == 200:
                logger.info("Playback started")
                return True
            return False
        except requests.RequestException as e:
            logger.error(f"Failed to start playback: {e}")
            return False

    def stop_playback(self) -> bool:
        """再生停止"""
        try:
            response = requests.post(
                f"{self.base_url}/stop",
                timeout=self.config.esp32_timeout
            )
            if response.status_code == 200:
                logger.info("Playback stopped")
                return True
            return False
        except requests.RequestException as e:
            logger.error(f"Failed to stop playback: {e}")
            return False

class VoiceTuner:
    """音声チューニングメインクラス"""

    def __init__(
        self,
        esp32_ip: str,
        target_audio_file: str,
        config: Optional[SystemConfig] = None
    ):
        self.config = config or SystemConfig()
        self.analyzer = AudioAnalyzer(self.config)
        self.controller = ESP32Controller(esp32_ip, self.config)
        self.target_audio_file = target_audio_file
        self.target_features: Optional[Dict] = None

        # 目標音声の解析
        self._analyze_target_audio()
    
    def _analyze_target_audio(self) -> None:
        """目標音声の特徴量抽出"""
        try:
            audio_data, sr = librosa.load(
                self.target_audio_file,
                sr=self.config.sample_rate
            )

            # 「あー」音の部分を抽出（音声全体から安定した部分を取得）
            # 簡単な音声活動検出
            rms = librosa.feature.rms(
                y=audio_data,
                frame_length=self.config.frame_size,
                hop_length=512
            )[0]
            rms_threshold = np.mean(rms) * 0.5

            # 安定した部分を抽出
            stable_frames = np.where(rms > rms_threshold)[0]
            if len(stable_frames) > 0:
                start_frame = stable_frames[0]
                end_frame = stable_frames[-1]
                start_sample = start_frame * 512
                end_sample = min(end_frame * 512 + self.config.frame_size, len(audio_data))
                stable_audio = audio_data[start_sample:end_sample]
            else:
                stable_audio = audio_data

            # 特徴量解析
            self.target_features = self.analyzer.analyze_spectrum(stable_audio)
            logger.info("目標音声解析完了:")
            logger.info(f"  基本周波数: {self.target_features['f0']:.1f} Hz")
            logger.info(f"  フォルマント: {[f'{f:.0f}' for f in self.target_features['formants']]} Hz")
            logger.info(f"  倍音構成: {[f'{h:.2f}' for h in self.target_features['harmonics']]}")

        except Exception as e:
            logger.error(f"目標音声の読み込みエラー: {e}")
            self.target_features = None
    
    def evaluate_similarity(self, current_features: Dict) -> float:
        """音声類似度評価"""
        if self.target_features is None:
            return 0.0

        score = 0.0

        # スペクトル類似度（コサイン類似度）
        target_spec = self.target_features['spectrum']
        current_spec = current_features['spectrum']

        # 長さを合わせる
        min_len = min(len(target_spec), len(current_spec))
        target_spec = target_spec[:min_len]
        current_spec = current_spec[:min_len]

        # コサイン類似度
        norm_product = np.linalg.norm(target_spec) * np.linalg.norm(current_spec)
        if norm_product > 1e-10:
            cosine_sim = np.dot(target_spec, current_spec) / norm_product
            score += cosine_sim * self.config.score_weight_spectrum

        # 倍音構成の類似度
        target_harmonics = np.array(self.target_features['harmonics'])
        current_harmonics = np.array(current_features['harmonics'])
        harmonic_diff = np.mean(np.abs(target_harmonics - current_harmonics))
        harmonic_score = max(0.0, 1.0 - harmonic_diff)
        score += harmonic_score * self.config.score_weight_harmonics

        # フォルマントの類似度（F1, F2のみ）
        target_formants = np.array(self.target_features['formants'][:2])
        current_formants = np.array(current_features['formants'][:2])
        if np.all(target_formants > 0):
            formant_diff = np.mean(np.abs(target_formants - current_formants) / target_formants)
            formant_score = max(0.0, 1.0 - formant_diff)
            score += formant_score * self.config.score_weight_formants

        return score


@dataclass
class Particle:
    """PSO用パーティクル"""
    rate: float
    amplitude: float
    harmonic_weights: List[float]
    velocity_rate: float = 0.0
    velocity_amplitude: float = 0.0
    velocity_harmonics: List[float] = field(default_factory=list)
    best_position: Optional[Dict] = None
    best_score: float = 0.0

    def __post_init__(self):
        if not self.velocity_harmonics:
            self.velocity_harmonics = [0.0] * len(self.harmonic_weights)

    def to_audio_parameters(self, config: SystemConfig) -> AudioParameters:
        """AudioParametersに変換"""
        return AudioParameters(
            playback_rate=self.rate,
            amplitude=self.amplitude,
            harmonic_weights=self.harmonic_weights.copy(),
            config=config
        )

    def get_position(self) -> Dict:
        """現在位置を辞書形式で取得"""
        return {
            'rate': self.rate,
            'amplitude': self.amplitude,
            'harmonic_weights': self.harmonic_weights.copy()
        }

    def update_best(self, score: float) -> None:
        """ベストスコア更新"""
        if score > self.best_score:
            self.best_score = score
            self.best_position = self.get_position()


class ParticleSwarmOptimizer:
    """粒子群最適化（PSO）実装"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.particles: List[Particle] = []
        self.global_best_position: Optional[Dict] = None
        self.global_best_score: float = 0.0

    def initialize_particles(self) -> None:
        """パーティクル初期化"""
        self.particles = []
        for _ in range(self.config.pso_n_particles):
            particle = Particle(
                rate=np.random.uniform(
                    self.config.rate_init_min,
                    self.config.rate_init_max
                ),
                amplitude=np.random.uniform(
                    self.config.amplitude_init_min,
                    self.config.amplitude_init_max
                ),
                harmonic_weights=np.random.uniform(
                    self.config.harmonic_weight_init_min,
                    self.config.harmonic_weight_init_max,
                    self.config.harmonic_count
                ).tolist(),
                velocity_rate=np.random.uniform(-0.1, 0.1),
                velocity_amplitude=np.random.uniform(-0.1, 0.1),
                velocity_harmonics=np.random.uniform(
                    -0.1, 0.1, self.config.harmonic_count
                ).tolist()
            )
            self.particles.append(particle)
        logger.info(f"Initialized {len(self.particles)} particles")

    def update_global_best(self, particle: Particle, score: float) -> None:
        """グローバルベスト更新"""
        if score > self.global_best_score:
            self.global_best_score = score
            self.global_best_position = particle.get_position()
            logger.info(f"New global best score: {score:.4f}")

    def update_particle_velocity_and_position(self, particle: Particle) -> None:
        """パーティクルの速度と位置を更新"""
        if self.global_best_position is None:
            return

        inertia = self.config.pso_inertia
        c1 = self.config.pso_c1
        c2 = self.config.pso_c2

        # 速度更新
        r1, r2 = np.random.random(), np.random.random()

        # 個人的ベストへの引力
        personal_attraction_rate = 0.0
        personal_attraction_amp = 0.0
        if particle.best_position:
            personal_attraction_rate = c1 * r1 * (
                particle.best_position['rate'] - particle.rate
            )
            personal_attraction_amp = c1 * r1 * (
                particle.best_position['amplitude'] - particle.amplitude
            )

        # グローバルベストへの引力
        global_attraction_rate = c2 * r2 * (
            self.global_best_position['rate'] - particle.rate
        )
        global_attraction_amp = c2 * r2 * (
            self.global_best_position['amplitude'] - particle.amplitude
        )

        # 速度更新
        particle.velocity_rate = (
            inertia * particle.velocity_rate +
            personal_attraction_rate +
            global_attraction_rate
        )
        particle.velocity_amplitude = (
            inertia * particle.velocity_amplitude +
            personal_attraction_amp +
            global_attraction_amp
        )

        # 倍音重みの速度更新
        for j in range(len(particle.harmonic_weights)):
            personal_attr = 0.0
            if particle.best_position:
                personal_attr = c1 * r1 * (
                    particle.best_position['harmonic_weights'][j] -
                    particle.harmonic_weights[j]
                )

            global_attr = c2 * r2 * (
                self.global_best_position['harmonic_weights'][j] -
                particle.harmonic_weights[j]
            )

            particle.velocity_harmonics[j] = (
                inertia * particle.velocity_harmonics[j] +
                personal_attr + global_attr
            )

        # 位置更新（境界制約付き）
        particle.rate = np.clip(
            particle.rate + particle.velocity_rate,
            self.config.rate_min,
            self.config.rate_max
        )
        particle.amplitude = np.clip(
            particle.amplitude + particle.velocity_amplitude,
            self.config.amplitude_min,
            self.config.amplitude_max
        )

        for j in range(len(particle.harmonic_weights)):
            particle.harmonic_weights[j] = np.clip(
                particle.harmonic_weights[j] + particle.velocity_harmonics[j],
                self.config.harmonic_weight_min,
                self.config.harmonic_weight_max
            )

    def update_all_particles(self) -> None:
        """全パーティクルの速度と位置を更新"""
        for particle in self.particles:
            self.update_particle_velocity_and_position(particle)

    def get_best_parameters(self) -> Optional[AudioParameters]:
        """最良パラメータを取得"""
        if self.global_best_position is None:
            return None

        return AudioParameters(
            playback_rate=self.global_best_position['rate'],
            amplitude=self.global_best_position['amplitude'],
            harmonic_weights=self.global_best_position['harmonic_weights'].copy(),
            config=self.config
        )


class VoiceTuner:
    """音声チューニングメインクラス"""

    def __init__(
        self,
        esp32_ip: str,
        target_audio_file: str,
        config: Optional[SystemConfig] = None
    ):
        self.config = config or SystemConfig()
        self.analyzer = AudioAnalyzer(self.config)
        self.controller = ESP32Controller(esp32_ip, self.config)
        self.target_audio_file = target_audio_file
        self.target_features: Optional[Dict] = None
        self.optimizer = ParticleSwarmOptimizer(self.config)

        # 目標音声の解析
        self._analyze_target_audio()

    def _analyze_target_audio(self) -> None:
        """目標音声の特徴量抽出"""
        try:
            audio_data, sr = librosa.load(
                self.target_audio_file,
                sr=self.config.sample_rate
            )

            # 「あー」音の部分を抽出（音声全体から安定した部分を取得）
            # 簡単な音声活動検出
            rms = librosa.feature.rms(
                y=audio_data,
                frame_length=self.config.frame_size,
                hop_length=512
            )[0]
            rms_threshold = np.mean(rms) * 0.5

            # 安定した部分を抽出
            stable_frames = np.where(rms > rms_threshold)[0]
            if len(stable_frames) > 0:
                start_frame = stable_frames[0]
                end_frame = stable_frames[-1]
                start_sample = start_frame * 512
                end_sample = min(end_frame * 512 + self.config.frame_size, len(audio_data))
                stable_audio = audio_data[start_sample:end_sample]
            else:
                stable_audio = audio_data

            # 特徴量解析
            self.target_features = self.analyzer.analyze_spectrum(stable_audio)
            logger.info("目標音声解析完了:")
            logger.info(f"  基本周波数: {self.target_features['f0']:.1f} Hz")
            logger.info(f"  フォルマント: {[f'{f:.0f}' for f in self.target_features['formants']]} Hz")
            logger.info(f"  倍音構成: {[f'{h:.2f}' for h in self.target_features['harmonics']]}")

        except Exception as e:
            logger.error(f"目標音声の読み込みエラー: {e}")
            self.target_features = None

    def evaluate_similarity(self, current_features: Dict) -> float:
        """音声類似度評価"""
        if self.target_features is None:
            return 0.0

        score = 0.0

        # スペクトル類似度（コサイン類似度）
        target_spec = self.target_features['spectrum']
        current_spec = current_features['spectrum']

        # 長さを合わせる
        min_len = min(len(target_spec), len(current_spec))
        target_spec = target_spec[:min_len]
        current_spec = current_spec[:min_len]

        # コサイン類似度
        norm_product = np.linalg.norm(target_spec) * np.linalg.norm(current_spec)
        if norm_product > 1e-10:
            cosine_sim = np.dot(target_spec, current_spec) / norm_product
            score += cosine_sim * self.config.score_weight_spectrum

        # 倍音構成の類似度
        target_harmonics = np.array(self.target_features['harmonics'])
        current_harmonics = np.array(current_features['harmonics'])
        harmonic_diff = np.mean(np.abs(target_harmonics - current_harmonics))
        harmonic_score = max(0.0, 1.0 - harmonic_diff)
        score += harmonic_score * self.config.score_weight_harmonics

        # フォルマントの類似度（F1, F2のみ）
        target_formants = np.array(self.target_features['formants'][:2])
        current_formants = np.array(current_features['formants'][:2])
        if np.all(target_formants > 0):
            formant_diff = np.mean(np.abs(target_formants - current_formants) / target_formants)
            formant_score = max(0.0, 1.0 - formant_diff)
            score += formant_score * self.config.score_weight_formants

        return score

    def _evaluate_particle(self, particle: Particle) -> float:
        """パーティクルの評価"""
        params = particle.to_audio_parameters(self.config)

        # パラメータをESP32に送信
        if not self.controller.send_parameters(params):
            logger.error("Failed to send parameters to ESP32")
            return 0.0

        # 再生開始
        if not self.controller.start_playback():
            logger.error("Failed to start playback")
            return 0.0

        # 安定化待ち
        time.sleep(self.config.stabilization_delay)

        # 録音・解析
        recorded_audio = self.analyzer.start_recording()
        self.controller.stop_playback()

        if len(recorded_audio) == 0:
            logger.warning("No audio recorded")
            return 0.0

        current_features = self.analyzer.analyze_spectrum(recorded_audio)
        score = self.evaluate_similarity(current_features)

        return score

    def optimize_parameters(
        self,
        max_iterations: Optional[int] = None
    ) -> AudioParameters:
        """パラメータ最適化"""
        if max_iterations is None:
            max_iterations = self.config.pso_max_iterations

        logger.info("パラメータ最適化開始...")
        logger.info(f"反復回数: {max_iterations}")
        logger.info(f"パーティクル数: {self.config.pso_n_particles}")

        # PSOの初期化
        self.optimizer.initialize_particles()

        # 最適化ループ
        for iteration in range(max_iterations):
            logger.info(f"\n=== 最適化反復 {iteration + 1}/{max_iterations} ===")

            # 各パーティクルを評価
            for i, particle in enumerate(self.optimizer.particles):
                score = self._evaluate_particle(particle)
                logger.info(f"  パーティクル {i+1}: スコア = {score:.4f}")

                # 個人ベスト更新
                particle.update_best(score)

                # グローバルベスト更新
                self.optimizer.update_global_best(particle, score)

                # 次のパーティクル評価前の小休止
                time.sleep(0.2)

            # 全パーティクルの速度と位置を更新
            self.optimizer.update_all_particles()

            logger.info(
                f"反復 {iteration + 1} 完了 - "
                f"現在のベストスコア: {self.optimizer.global_best_score:.4f}"
            )

        # 最良パラメータを取得
        best_params = self.optimizer.get_best_parameters()

        if best_params is None:
            logger.error("最適化に失敗しました")
            return AudioParameters(config=self.config)

        logger.info(f"\n最適化完了! 最高スコア: {self.optimizer.global_best_score:.4f}")
        logger.info("最適パラメータ:")
        logger.info(f"  再生速度: {best_params.playback_rate:.3f}")
        logger.info(f"  音量: {best_params.amplitude:.3f}")
        logger.info(f"  倍音構成: {[f'{w:.3f}' for w in best_params.harmonic_weights]}")

        return best_params
    
    def save_parameters(self, params: AudioParameters, filename: str) -> None:
        """パラメータをファイルに保存"""
        try:
            data = params.to_dict()

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"パラメータを {filename} に保存しました")
        except Exception as e:
            logger.error(f"パラメータ保存エラー: {e}")

def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='人工咽頭チューニングシステム',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--esp32-ip',
        required=True,
        help='ESP32のIPアドレス'
    )
    parser.add_argument(
        '--target-audio',
        required=True,
        help='目標音声ファイル（WAV/MP3）'
    )
    parser.add_argument(
        '--output',
        default='optimized_params.json',
        help='出力パラメータファイル'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='最適化反復回数（デフォルト: 設定値）'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='ログレベル'
    )

    args = parser.parse_args()

    # ログレベル設定
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # チューニングシステム初期化
        logger.info("システム初期化中...")
        config = SystemConfig()
        if args.iterations is not None:
            config.pso_max_iterations = args.iterations

        tuner = VoiceTuner(args.esp32_ip, args.target_audio, config)

        if tuner.target_features is None:
            logger.error("目標音声の読み込みに失敗しました")
            return

        # 最適化実行
        optimal_params = tuner.optimize_parameters()

        # 結果保存
        tuner.save_parameters(optimal_params, args.output)

        # 最終確認
        logger.info("\n最終確認用再生...")
        tuner.controller.send_parameters(optimal_params)
        tuner.controller.start_playback()

        try:
            input("再生中... Enterキーで停止")
        finally:
            tuner.controller.stop_playback()

        logger.info("チューニング完了")

    except KeyboardInterrupt:
        logger.info("\n最適化を中断しました")
        try:
            tuner.controller.stop_playback()
        except:
            pass
    except Exception as e:
        logger.exception(f"エラーが発生しました: {e}")
        try:
            tuner.controller.stop_playback()
        except:
            pass


if __name__ == "__main__":
    main()