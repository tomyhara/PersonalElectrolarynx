# Personal Electrolarynx

# チューニングシステム セットアップガイド

## 必要な環境

### ハードウェア
- PC（Windows/Mac/Linux）
- マイク（内蔵またはUSBマイク）
- ESP32-S3 + MAX98357A + 振動スピーカー

### ソフトウェア
- Python 3.8以上
- PlatformIO（ESP32開発用）

## セットアップ手順

### 1. Python環境の準備

```bash
# 仮想環境作成（推奨）
python -m venv voice_tuning_env

# 仮想環境アクティベート
# Windows:
voice_tuning_env\Scripts\activate
# Mac/Linux:
source voice_tuning_env/bin/activate

# 必要ライブラリのインストール
pip install -r requirements.txt
```

### 2. ESP32側の準備

#### ESP32プログラムの修正点
チューニング用にESP32側のプログラムに以下の機能を追加：

```cpp
// main.cppに追加
void handleUpdateHarmonics() {
  if (server.hasArg("plain")) {
    DynamicJsonDocument doc(2048);
    deserializeJson(doc, server.arg("plain"));
    
    playbackRate = doc["rate"];
    amplitudeScale = doc["amplitude"];
    
    // 倍音構成の更新
    JsonArray harmonics = doc["harmonics"];
    if (harmonics.size() >= 5) {
      for (int i = 0; i < 5; i++) {
        harmonicWeights[i] = harmonics[i];
      }
      generateAdvancedWaveform();  // 新しい波形生成関数
    }
    
    server.send(200, "text/plain", "Harmonics Updated");
  } else {
    server.send(400, "text/plain", "Bad Request");
  }
}

// setup()内に追加
server.on("/update_harmonics", HTTP_POST, handleUpdateHarmonics);
```

### 3. 音声ファイルの準備

目標となる「あー」音声ファイルを準備：

```bash
# 推奨フォーマット
- ファイル形式: WAV (16bit, 22050Hz) または MP3
- 長さ: 2-5秒
- 内容: 安定した「あー」音（できるだけノイズなし）
- ファイル名例: target_voice.wav
```

## 使用方法

### 基本的な使用手順

```bash
# 1. ESP32のIPアドレスを確認
# ESP32のシリアルモニターまたはルーターの管理画面で確認

# 2. チューニング実行
python voice_tuner.py --esp32-ip 192.168.1.100 --target-audio target_voice.wav

# 3. オプション指定
python voice_tuner.py \
  --esp32-ip 192.168.1.100 \
  --target-audio target_voice.wav \
  --output my_optimized_params.json \
  --iterations 30
```

### パラメータ説明

- `--esp32-ip`: ESP32のIPアドレス
- `--target-audio`: 目標音声ファイルのパス
- `--output`: 最適化結果の保存ファイル名（デフォルト: optimized_params.json）
- `--iterations`: 最適化の反復回数（デフォルト: 20）

## チューニングプロセス

### 1. 準備段階
- 目標音声の解析（基本周波数、倍音構成、フォルマント）
- ESP32との通信確認

### 2. 最適化段階
- 粒子群最適化（PSO）による自動パラメータ探索
- 各パラメータ組み合わせでの音声生成・録音・評価
- リアルタイム進捗表示

### 3. 結果保存
- 最適パラメータのJSON形式保存
- ESP32への最終パラメータ適用

## トラブルシューティング

### 音声録音の問題
```python
# 使用可能なオーディオデバイス確認
import sounddevice as sd
print(sd.query_devices())

# 特定のデバイスを指定する場合
sd.default.device = 'your_device_name'
```

### ESP32通信の問題
- WiFi接続の確認
- IPアドレスの再確認
- ファイアウォール設定の確認

### 最適化が収束しない場合
- 反復回数を増やす（--iterations 50）
- 目標音声の品質を確認（ノイズ除去）
- マイクの位置・距離を調整

## 高度な使用法

### カスタム評価関数
```python
# voice_tuner.pyを修正して独自の評価指標を追加
def custom_evaluate_similarity(self, current_features: dict) -> float:
    # 独自の類似度計算ロジック
    pass
```

### バッチ処理
```python
# 複数の目標音声で一括最適化
target_files = ['voice1.wav', 'voice2.wav', 'voice3.wav']
for target in target_files:
    # 各音声でチューニング実行
```

## 結果の活用

最適化完了後、`optimized_params.json`をESP32に適用：

1. **手動適用**: WebUIでパラメータを手動入力
2. **自動適用**: ESP32側でJSON読み込み機能を実装
3. **製品化**: 最適パラメータを固定値としてプログラムに組み込み