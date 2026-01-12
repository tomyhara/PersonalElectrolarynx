# 人工咽頭 Voice Synthesizer ハードウェア

ESP32-S3ベースの音声合成システムです。振動スピーカーを使用して、喉に当てることで音声を生成します。

## 目次

- [概要](#概要)
- [必要な部品](#必要な部品)
- [配線](#配線)
- [セットアップ](#セットアップ)
- [ビルドと書き込み](#ビルドと書き込み)
- [使用方法](#使用方法)
- [WiFi設定](#wifi設定)
- [API仕様](#api仕様)
- [トラブルシューティング](#トラブルシューティング)

## 概要

このシステムは以下の機能を提供します：

- **リアルタイム音声合成**: 倍音構成とエンベロープを使用した「あー」音の生成
- **WebUI制御**: ブラウザから再生速度・音量・倍音構成を調整可能
- **Python連携**: チューニングシステムとHTTP APIで通信
- **ボタン制御**: 物理ボタンで音声のオン/オフ切り替え
- **I2S出力**: MAX98357A DACアンプ経由で高品質オーディオ出力

## 必要な部品

### マイコンボード
- **ESP32-S3 DevKitC-1** (またはESP32-S3互換ボード)
  - PSRAMモデル推奨
  - USB接続可能なもの

### オーディオ関連
- **MAX98357A I2S DACアンプモジュール**
  - 3.3V動作
  - I2Sインターフェース
- **振動スピーカー** (または小型スピーカー)
  - 8Ω、3-5W程度
  - 推奨: 振動トランスデューサー

### その他
- **タクトスイッチ** (1個)
  - 発声トリガー用
- **LED** (1個)
  - 状態表示用 (任意)
- **抵抗** 220Ω〜1kΩ (LED用、任意)
- **ジャンパーワイヤー** 適量
- **ブレッドボード** (プロトタイピング用)

## 配線

### ピン配置

| ESP32-S3 | 接続先 | 説明 |
|----------|--------|------|
| GPIO42 | MAX98357A BCLK | I2S ビットクロック |
| GPIO2 | MAX98357A LRCLK | I2S LR クロック (WS) |
| GPIO41 | MAX98357A DIN | I2S データ入力 |
| GPIO0 | タクトスイッチ | 発声トリガーボタン (プルアップ) |
| GPIO48 | LED (+ 抵抗) | 状態表示LED |
| 3.3V | MAX98357A VIN | 電源 |
| GND | MAX98357A GND | グラウンド |

### 配線図

```
ESP32-S3 DevKitC-1          MAX98357A
┌─────────────┐             ┌──────────┐
│             │             │          │
│    GPIO42───┼────────────►│ BCLK     │
│    GPIO2────┼────────────►│ LRCLK(WS)│
│    GPIO41───┼────────────►│ DIN      │
│             │             │          │
│    3.3V─────┼────────────►│ VIN      │
│    GND──────┼────────────►│ GND      │
│             │             │          │
│             │             │ SPKR+ ───┼──┐
│             │             │ SPKR- ───┼──┤ 振動スピーカー
│             │             └──────────┘  │  (8Ω)
│             │                           │
│             │                           │
│    GPIO0────┼───┬─ タクトスイッチ ─ GND  │
│             │   │                       │
│   GPIO48────┼───┴─ LED ─ 抵抗 ─ GND    │
│             │       (220Ω)              │
└─────────────┘                           │
                                          │
```

### 配線の注意点

1. **MAX98357Aの電源**
   - VINには3.3Vを接続 (5Vでも動作するが3.3V推奨)
   - GNDを確実に接続

2. **スピーカー接続**
   - SPKR+とSPKR-にスピーカーを接続
   - 極性は基本的に関係ないが、音質が気になる場合は試行

3. **ボタン配線**
   - GPIO0は内部プルアップを使用
   - ボタンでGNDに接続（押下時LOW）

## セットアップ

### 1. 開発環境の準備

#### PlatformIO CLI のインストール

```bash
# Python環境がある場合
pip install platformio

# または、VSCode拡張機能を使用
# VSCode Extensions > PlatformIO IDE をインストール
```

#### VS Codeを使用する場合

1. VS Codeをインストール
2. PlatformIO IDE拡張機能をインストール
3. このプロジェクトフォルダを開く

### 2. 依存ライブラリ

`platformio.ini`に記載されており、自動的にインストールされます：

- **ArduinoJson** ^6.21.3 - JSON処理用

## ビルドと書き込み

### PlatformIO CLI の場合

```bash
# プロジェクトディレクトリに移動
cd hardware/voice_synthesizer

# ビルド
pio run

# ESP32に書き込み
pio run --target upload

# シリアルモニター起動
pio device monitor
```

### VS Code (PlatformIO) の場合

1. PlatformIOサイドバーを開く
2. `Project Tasks` > `esp32-s3-devkitc-1` を展開
3. `Build` をクリックしてビルド
4. `Upload` をクリックして書き込み
5. `Monitor` をクリックしてシリアル出力を確認

### ビルド設定の説明

- **CPU周波数**: 240MHz
- **フラッシュ周波数**: 80MHz
- **パーティション**: `huge_app.csv` (大きなアプリ用)
- **PSRAMサポート**: 有効

## 使用方法

### 1. 起動

ESP32に電源を供給すると、以下の初期化シーケンスが実行されます：

```
========================================
人工咽頭システム起動中...
========================================
[1/7] GPIO設定中...
  GPIO設定完了
[2/7] SPIFFS初期化中...
  SPIFFS初期化完了
[3/7] I2Sオーディオ初期化中...
  I2S初期化成功
[4/7] 波形テーブル生成中...
  波形生成完了
[5/7] WiFi接続中...
[6/7] Webサーバー起動中...
  Webサーバー起動完了
[7/7] オーディオタスク起動中...
  オーディオタスク起動完了

========================================
システム準備完了!
========================================
WebUI: http://192.168.4.1
AP SSID: ESP32-VoiceSynth
========================================
```

### 2. 物理ボタンで制御

- **GPIO0のボタンを押す**: 音声再生の開始/停止を切り替え
- **LED (GPIO48)**: 再生中は点灯、停止中は消灯

### 3. WebUIで制御

1. スマートフォンまたはPCでWiFiに接続
   - SSID: `ESP32-VoiceSynth`
   - パスワード: `12345678`

2. ブラウザで `http://192.168.4.1` にアクセス

3. WebUI上で以下を調整可能：
   - **再生速度**: 0.5〜2.0 (ピッチ変更)
   - **音量**: 0.1〜1.0
   - **波形更新**: パラメータを適用

## WiFi設定

### APモード (デフォルト)

デフォルトではアクセスポイントモードで起動します：

- **SSID**: `ESP32-VoiceSynth`
- **パスワード**: `12345678`
- **IPアドレス**: `192.168.4.1`

### STAモード (既存WiFiに接続)

既存のWiFiネットワークに接続する場合は、`src/main.cpp`の`setupWiFi()`関数を変更してください：

```cpp
void setupWiFi() {
  // APモードの代わりにSTAモードを使用
  WiFi.begin("YourWiFiSSID", "YourWiFiPassword");

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}
```

## API仕様

Pythonチューニングシステムとの連携用HTTP API

### エンドポイント

#### 1. ルート - WebUI

```
GET /
```

HTML形式のWebUIを返します。

---

#### 2. 再生開始

```
POST /play
```

音声の再生を開始します。

**レスポンス**:
```
200 OK
Playing
```

---

#### 3. 再生停止

```
POST /stop
```

音声の再生を停止します。

**レスポンス**:
```
200 OK
Stopped
```

---

#### 4. パラメータ更新

```
POST /update
Content-Type: application/json
```

音声パラメータを更新します。

**リクエストボディ**:
```json
{
  "rate": 1.0,
  "amplitude": 0.8,
  "harmonics": [1.0, 0.4, 0.2, 0.1, 0.05]
}
```

**パラメータ説明**:
- `rate` (float): 再生速度 (0.5〜2.0)
- `amplitude` (float): 音量 (0.1〜1.0)
- `harmonics` (array): 倍音重み配列 (5要素)

**レスポンス**:
```
200 OK
Updated
```

**エラーレスポンス**:
```
400 Bad Request
Bad Request: No JSON body
```

---

### API使用例 (Python)

```python
import requests

# ESP32のIPアドレス
esp32_ip = "192.168.4.1"  # APモードの場合

# パラメータ更新
params = {
    "rate": 1.2,
    "amplitude": 0.75,
    "harmonics": [1.0, 0.5, 0.3, 0.15, 0.08]
}

response = requests.post(
    f"http://{esp32_ip}/update",
    json=params,
    timeout=5
)

if response.status_code == 200:
    print("パラメータ更新成功")

# 再生開始
requests.post(f"http://{esp32_ip}/play")

# 3秒後に停止
import time
time.sleep(3)
requests.post(f"http://{esp32_ip}/stop")
```

## トラブルシューティング

### 1. 書き込みに失敗する

**症状**: `Failed to connect to ESP32`

**解決方法**:
- USB接続を確認
- ボタンを押しながらリセットしてブートローダーモードに入る
- 別のUSBケーブルを試す
- ドライバーが正しくインストールされているか確認 (CP210x, CH340)

---

### 2. 音が出ない

**症状**: LEDは点灯するが音声が出力されない

**解決方法**:
1. **配線を確認**
   - MAX98357AのBCLK, LRCLK, DINが正しく接続されているか
   - スピーカーがSPKR+, SPKR-に接続されているか

2. **シリアル出力を確認**
   ```bash
   pio device monitor
   ```
   - `I2S初期化成功` が表示されているか確認

3. **スピーカーを確認**
   - 別のスピーカーで試す
   - 8Ω、3W以上のスピーカーを使用

4. **MAX98357Aの電源**
   - VINに3.3Vが供給されているか
   - GNDが接続されているか

---

### 3. WiFiに接続できない

**症状**: スマホ/PCがESP32のAPに接続できない

**解決方法**:
- SSID `ESP32-VoiceSynth` が表示されているか確認
- パスワード `12345678` を正しく入力
- ESP32を再起動してみる
- シリアルモニターでWiFi状態を確認

---

### 4. WebUIにアクセスできない

**症状**: `http://192.168.4.1` が開けない

**解決方法**:
- WiFi接続を確認 (APモードの場合)
- シリアルモニターでIPアドレスを確認
- ブラウザのキャッシュをクリア
- 別のブラウザで試す

---

### 5. 音がノイズまみれ・歪む

**症状**: 音声が歪んでいる、ノイズが多い

**解決方法**:
1. **振幅を下げる**
   - WebUIで音量を0.5以下に設定
   - `amplitudeScale` の初期値を0.5に変更

2. **電源を確認**
   - USB電源の品質を確認
   - 別の電源アダプターを試す

3. **配線の長さ**
   - I2S信号線を短くする (10cm以下推奨)
   - ツイストペア線を使用

4. **グラウンドを共通化**
   - ESP32とMAX98357AのGNDが確実に接続されているか

---

### 6. Pythonチューニングシステムと通信できない

**症状**: Pythonから `/update` APIが失敗する

**解決方法**:
1. **ネットワーク接続確認**
   ```bash
   ping 192.168.4.1
   ```

2. **シリアルログを確認**
   - `Error: JSON parse failed` が出ていないか
   - `Parameters updated` が表示されるか

3. **JSONフォーマット確認**
   ```python
   # 正しいフォーマット
   {
     "rate": 1.0,
     "amplitude": 0.8,
     "harmonics": [1.0, 0.4, 0.2, 0.1, 0.05]
   }
   ```

## カスタマイズ

### 定数の変更

`src/main.cpp`の冒頭で定義されている定数を変更することで、動作をカスタマイズできます：

```cpp
// サンプリングレート変更 (22050 → 44100)
#define SAMPLE_RATE 44100

// 倍音の数を増やす (5 → 10)
#define NUM_HARMONICS 10

// アタック時間を短く (0.1 → 0.05)
#define ATTACK_TIME 0.05f
```

### WiFi設定のカスタマイズ

```cpp
// APモードのSSID/パスワード変更
#define WIFI_AP_SSID "MyVoiceSynth"
#define WIFI_AP_PASSWORD "mypassword123"
```

## ライセンス

このプロジェクトのライセンスについては、ルートディレクトリの`LICENSE`ファイルを参照してください。

## 関連ドキュメント

- [プロジェクトルートREADME](../../README.md)
- [Pythonチューニングシステム](../../src/tuner/README.md) (予定)
- [MAX98357Aデータシート](https://www.analog.com/media/en/technical-documentation/data-sheets/MAX98357A-MAX98357B.pdf)
- [ESP32-S3技術リファレンス](https://www.espressif.com/sites/default/files/documentation/esp32-s3_technical_reference_manual_en.pdf)

## サポート

問題が発生した場合は、GitHubのIssueで報告してください。
