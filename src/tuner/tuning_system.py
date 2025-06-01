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
from dataclasses import dataclass
from typing import Tuple, List, Optional
import argparse

@dataclass
class AudioParameters:
    """音声パラメータクラス"""
    playback_rate: float = 1.0
    amplitude: float = 0.8
    formant_f1: float = 700.0  # 第1フォルマント
    formant_f2: float = 1200.0  # 第2フォルマント
    harmonic_weights: List[float] = None
    
    def __post_init__(self):
        if self.harmonic_weights is None:
            # デフォルトの倍音構成（あー音の近似）
            self.harmonic_weights = [1.0, 0.4, 0.2, 0.1, 0.05]

class AudioAnalyzer:
    """音声解析クラス"""
    
    def __init__(self, sample_rate=22050, frame_size=2048):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.audio_queue = queue.Queue()
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
        # 100-400Hzの範囲で最大ピークを探す
        mask = (freqs >= 80) & (freqs <= 400)
        masked_spectrum = spectrum.copy()
        masked_spectrum[~mask] = 0
        
        peak_idx = np.argmax(masked_spectrum)
        return freqs[peak_idx] if peak_idx > 0 else 150.0
    
    def _analyze_harmonics(self, freqs: np.ndarray, spectrum: np.ndarray, f0: float) -> List[float]:
        """倍音解析"""
        harmonics = []
        for i in range(1, 6):  # 1-5倍音
            harmonic_freq = f0 * i
            # ±10Hzの範囲で最大値を探す
            mask = (freqs >= harmonic_freq - 10) & (freqs <= harmonic_freq + 10)
            if np.any(mask):
                harmonic_amplitude = np.max(spectrum[mask])
                harmonics.append(harmonic_amplitude)
            else:
                harmonics.append(0.0)
        
        # 基本波で正規化
        if harmonics[0] > 0:
            harmonics = [h / harmonics[0] for h in harmonics]
        
        return harmonics
    
    def _detect_formants(self, freqs: np.ndarray, spectrum: np.ndarray) -> List[float]:
        """フォルマント検出"""
        # スペクトルエンベロープを求める
        envelope = signal.savgol_filter(spectrum, 51, 3)
        
        # ピーク検出
        peaks, _ = signal.find_peaks(envelope, height=np.max(envelope) * 0.1, distance=50)
        
        formant_freqs = []
        for peak in peaks[:3]:  # 上位3つのフォルマント
            formant_freqs.append(freqs[peak])
        
        # 足りない場合はデフォルト値で埋める
        defaults = [700, 1200, 2500]
        while len(formant_freqs) < 3:
            formant_freqs.append(defaults[len(formant_freqs)])
        
        return formant_freqs[:3]
    
    def start_recording(self, duration: float = 2.0):
        """録音開始"""
        def record_callback(indata, frames, time, status):
            if status:
                print(f"Recording status: {status}")
            self.audio_queue.put(indata.copy())
        
        self.recording = True
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=record_callback,
            blocksize=self.frame_size
        ):
            print(f"Recording for {duration} seconds...")
            time.sleep(duration)
        
        self.recording = False
        
        # 録音データを結合
        audio_blocks = []
        while not self.audio_queue.empty():
            audio_blocks.append(self.audio_queue.get())
        
        if audio_blocks:
            return np.concatenate(audio_blocks, axis=0).flatten()
        else:
            return np.array([])

class ESP32Controller:
    """ESP32制御クラス"""
    
    def __init__(self, esp32_ip: str):
        self.esp32_ip = esp32_ip
        self.base_url = f"http://{esp32_ip}"
        
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
                timeout=5
            )
            return response.status_code == 200
            
        except requests.RequestException as e:
            print(f"ESP32通信エラー: {e}")
            return False
    
    def start_playback(self) -> bool:
        """再生開始"""
        try:
            response = requests.post(f"{self.base_url}/play", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def stop_playback(self) -> bool:
        """再生停止"""
        try:
            response = requests.post(f"{self.base_url}/stop", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

class VoiceTuner:
    """音声チューニングメインクラス"""
    
    def __init__(self, esp32_ip: str, target_audio_file: str):
        self.analyzer = AudioAnalyzer()
        self.controller = ESP32Controller(esp32_ip)
        self.target_audio_file = target_audio_file
        self.target_features = None
        
        # 目標音声の解析
        self._analyze_target_audio()
    
    def _analyze_target_audio(self):
        """目標音声の特徴量抽出"""
        try:
            audio_data, sr = librosa.load(self.target_audio_file, sr=22050)
            
            # 「あー」音の部分を抽出（音声全体から安定した部分を取得）
            # 簡単な音声活動検出
            rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
            rms_threshold = np.mean(rms) * 0.5
            
            # 安定した部分を抽出
            stable_frames = np.where(rms > rms_threshold)[0]
            if len(stable_frames) > 0:
                start_frame = stable_frames[0]
                end_frame = stable_frames[-1]
                start_sample = start_frame * 512
                end_sample = min(end_frame * 512 + 2048, len(audio_data))
                stable_audio = audio_data[start_sample:end_sample]
            else:
                stable_audio = audio_data
            
            # 特徴量解析
            self.target_features = self.analyzer.analyze_spectrum(stable_audio)
            print(f"目標音声解析完了:")
            print(f"  基本周波数: {self.target_features['f0']:.1f} Hz")
            print(f"  フォルマント: {[f'{f:.0f}' for f in self.target_features['formants']]} Hz")
            print(f"  倍音構成: {[f'{h:.2f}' for h in self.target_features['harmonics']]}")
            
        except Exception as e:
            print(f"目標音声の読み込みエラー: {e}")
            self.target_features = None
    
    def evaluate_similarity(self, current_features: dict) -> float:
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
        cosine_sim = np.dot(target_spec, current_spec) / (
            np.linalg.norm(target_spec) * np.linalg.norm(current_spec) + 1e-10
        )
        score += cosine_sim * 0.4
        
        # 倍音構成の類似度
        target_harmonics = np.array(self.target_features['harmonics'])
        current_harmonics = np.array(current_features['harmonics'])
        harmonic_diff = np.mean(np.abs(target_harmonics - current_harmonics))
        harmonic_score = max(0, 1 - harmonic_diff)
        score += harmonic_score * 0.4
        
        # フォルマントの類似度
        target_formants = np.array(self.target_features['formants'][:2])  # F1, F2のみ
        current_formants = np.array(current_features['formants'][:2])
        formant_diff = np.mean(np.abs(target_formants - current_formants) / target_formants)
        formant_score = max(0, 1 - formant_diff)
        score += formant_score * 0.2
        
        return score
    
    def optimize_parameters(self, max_iterations: int = 20) -> AudioParameters:
        """パラメータ最適化"""
        print("パラメータ最適化開始...")
        
        best_params = AudioParameters()
        best_score = 0.0
        
        # 粒子群最適化（PSO）の簡易実装
        n_particles = 8
        particles = []
        velocities = []
        
        # パーティクル初期化
        for _ in range(n_particles):
            particle = {
                'rate': np.random.uniform(0.8, 1.2),
                'amplitude': np.random.uniform(0.5, 1.0),
                'harmonic_weights': np.random.uniform(0.1, 1.0, 5).tolist()
            }
            particles.append(particle)
            
            velocity = {
                'rate': np.random.uniform(-0.1, 0.1),
                'amplitude': np.random.uniform(-0.1, 0.1),
                'harmonic_weights': np.random.uniform(-0.1, 0.1, 5).tolist()
            }
            velocities.append(velocity)
        
        global_best = None
        global_best_score = 0.0
        
        for iteration in range(max_iterations):
            print(f"\n最適化反復 {iteration + 1}/{max_iterations}")
            
            for i, particle in enumerate(particles):
                # パラメータをESP32に送信
                params = AudioParameters(
                    playback_rate=particle['rate'],
                    amplitude=particle['amplitude'],
                    harmonic_weights=particle['harmonic_weights']
                )
                
                if not self.controller.send_parameters(params):
                    print("ESP32への送信失敗")
                    continue
                
                # 再生開始
                self.controller.start_playback()
                time.sleep(0.5)  # 安定待ち
                
                # 録音・解析
                recorded_audio = self.analyzer.start_recording(1.5)
                if len(recorded_audio) > 0:
                    current_features = self.analyzer.analyze_spectrum(recorded_audio)
                    score = self.evaluate_similarity(current_features)
                    
                    print(f"  パーティクル {i+1}: スコア = {score:.3f}")
                    
                    # 最良解の更新
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    if score > global_best_score:
                        global_best_score = score
                        global_best = particle.copy()
                
                # 再生停止
                self.controller.stop_playback()
                time.sleep(0.2)
            
            # パーティクル更新（簡易PSO）
            for i, (particle, velocity) in enumerate(zip(particles, velocities)):
                if global_best is not None:
                    # 速度更新
                    inertia = 0.5
                    c1, c2 = 1.5, 1.5  # 学習係数
                    
                    velocity['rate'] = (inertia * velocity['rate'] + 
                                      c1 * np.random.random() * (global_best['rate'] - particle['rate']))
                    velocity['amplitude'] = (inertia * velocity['amplitude'] + 
                                           c2 * np.random.random() * (global_best['amplitude'] - particle['amplitude']))
                    
                    for j in range(5):
                        velocity['harmonic_weights'][j] = (
                            inertia * velocity['harmonic_weights'][j] + 
                            c1 * np.random.random() * (global_best['harmonic_weights'][j] - particle['harmonic_weights'][j])
                        )
                    
                    # 位置更新
                    particle['rate'] = np.clip(particle['rate'] + velocity['rate'], 0.5, 2.0)
                    particle['amplitude'] = np.clip(particle['amplitude'] + velocity['amplitude'], 0.1, 1.0)
                    
                    for j in range(5):
                        particle['harmonic_weights'][j] = np.clip(
                            particle['harmonic_weights'][j] + velocity['harmonic_weights'][j], 
                            0.01, 1.0
                        )
        
        print(f"\n最適化完了! 最高スコア: {best_score:.3f}")
        print(f"最適パラメータ:")
        print(f"  再生速度: {best_params.playback_rate:.3f}")
        print(f"  音量: {best_params.amplitude:.3f}")
        print(f"  倍音構成: {[f'{w:.3f}' for w in best_params.harmonic_weights]}")
        
        return best_params
    
    def save_parameters(self, params: AudioParameters, filename: str):
        """パラメータをファイルに保存"""
        data = {
            'playback_rate': params.playback_rate,
            'amplitude': params.amplitude,
            'harmonic_weights': params.harmonic_weights,
            'formant_f1': params.formant_f1,
            'formant_f2': params.formant_f2
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"パラメータを {filename} に保存しました")

def main():
    parser = argparse.ArgumentParser(description='人工咽頭チューニングシステム')
    parser.add_argument('--esp32-ip', required=True, help='ESP32のIPアドレス')
    parser.add_argument('--target-audio', required=True, help='目標音声ファイル（WAV/MP3）')
    parser.add_argument('--output', default='optimized_params.json', help='出力パラメータファイル')
    parser.add_argument('--iterations', type=int, default=20, help='最適化反復回数')
    
    args = parser.parse_args()
    
    # チューニングシステム初期化
    tuner = VoiceTuner(args.esp32_ip, args.target_audio)
    
    if tuner.target_features is None:
        print("目標音声の読み込みに失敗しました")
        return
    
    try:
        # 最適化実行
        optimal_params = tuner.optimize_parameters(args.iterations)
        
        # 結果保存
        tuner.save_parameters(optimal_params, args.output)
        
        # 最終確認
        print("\n最終確認用再生...")
        tuner.controller.send_parameters(optimal_params)
        tuner.controller.start_playback()
        
        input("再生中... Enterキーで停止")
        tuner.controller.stop_playback()
        
    except KeyboardInterrupt:
        print("\n最適化を中断しました")
        tuner.controller.stop_playback()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        tuner.controller.stop_playback()

if __name__ == "__main__":
    main()