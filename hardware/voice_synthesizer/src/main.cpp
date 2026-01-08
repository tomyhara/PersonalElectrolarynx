#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>
#include <ArduinoJson.h>

// ================================================================================
// ピン定義
// ================================================================================
#define I2S_BCLK_PIN    42  // I2S BCK (Bit Clock) pin
#define I2S_LRCK_PIN    2   // I2S LRCK (Left/Right Clock) pin
#define I2S_DATA_PIN    41  // I2S DIN (Data In) pin
#define BUTTON_PIN      0   // 発声トリガーボタンピン
#define LED_PIN         48  // 状態表示LEDピン

// ================================================================================
// 音声パラメータ定数
// ================================================================================
#define SAMPLE_RATE         22050   // サンプリングレート (Hz)
#define WAVE_TABLE_SIZE     1024    // 波形テーブルサイズ
#define AMPLITUDE_MAX       16383   // 最大振幅 (16bit signedの半分)
#define AUDIO_BUFFER_SIZE   128     // オーディオバッファサイズ
#define TASK_STACK_SIZE     4096    // オーディオタスクのスタックサイズ
#define TASK_PRIORITY       1       // オーディオタスクの優先度
#define TASK_CORE           1       // オーディオタスクを実行するCPUコア

// ================================================================================
// 波形生成パラメータ
// ================================================================================
#define NUM_HARMONICS       5       // 倍音の数
#define ATTACK_TIME         0.1f    // アタック時間（波形テーブルの比率）
#define RELEASE_TIME        0.2f    // リリース時間（波形テーブルの比率）
#define RELEASE_START       0.8f    // リリース開始位置（波形テーブルの比率）

// ================================================================================
// WiFi/Webサーバー設定
// ================================================================================
#define WEB_SERVER_PORT     80      // Webサーバーのポート
#define WIFI_AP_SSID        "ESP32-VoiceSynth"  // APモードのSSID
#define WIFI_AP_PASSWORD    "12345678"          // APモードのパスワード
#define JSON_BUFFER_SIZE    1024    // JSON処理バッファサイズ

// ================================================================================
// グローバル変数
// ================================================================================
// 再生状態
volatile bool isPlaying = false;
volatile bool buttonPressed = false;

// 音声パラメータ
float playbackRate = 1.0f;
float amplitudeScale = 0.8f;
float harmonicWeights[NUM_HARMONICS] = {1.0f, 0.4f, 0.2f, 0.1f, 0.05f};  // デフォルト倍音構成

// 波形テーブル
int16_t waveTable[WAVE_TABLE_SIZE];

// Webサーバー
WebServer server(WEB_SERVER_PORT);

// ================================================================================
// I2S設定関数
// ================================================================================
/**
 * I2Sインターフェースを初期化
 * MAX98357Aアンプモジュールとの通信を設定
 * @return true: 成功, false: 失敗
 */
bool setupI2S() {
  // I2S設定構造体
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),  // マスターモード、送信のみ
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,          // モノラル（左チャンネルのみ）
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,             // 割り込みフラグ
    .dma_buf_count = 8,                                    // DMAバッファ数
    .dma_buf_len = 64,                                     // DMAバッファ長
    .use_apll = false,                                     // APLL不使用
    .tx_desc_auto_clear = true,                           // DMA descriptor自動クリア
    .fixed_mclk = 0
  };

  // I2Sピン設定
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCLK_PIN,
    .ws_io_num = I2S_LRCK_PIN,
    .data_out_num = I2S_DATA_PIN,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  // I2Sドライバのインストール
  esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("I2Sドライバのインストールに失敗: %d\n", err);
    return false;
  }

  // I2Sピンの設定
  err = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("I2Sピン設定に失敗: %d\n", err);
    return false;
  }

  Serial.println("I2S初期化成功");
  return true;
}

// ================================================================================
// 波形生成関数
// ================================================================================
/**
 * 基本波形テーブルを生成
 * 倍音構成とエンベロープを使用して「あー」音を合成
 */
void generateBaseWaveform() {
  for (int i = 0; i < WAVE_TABLE_SIZE; i++) {
    float t = (float)i / WAVE_TABLE_SIZE;  // 正規化時間 [0.0, 1.0]

    // 倍音合成による波形生成
    float wave = 0.0f;
    for (int harmonic = 0; harmonic < NUM_HARMONICS; harmonic++) {
      // 各倍音成分を加算（基本波 + N倍音）
      wave += harmonicWeights[harmonic] * sin(2.0f * PI * t * (harmonic + 1));
    }

    // フォルマント風の特性を追加（音声らしさの向上）
    float formant = 0.3f * sin(2.0f * PI * t * 8.0f) * exp(-t * 2.0f);
    wave += formant;

    // エンベロープ（ADSR風：アタック・サステイン・リリース）
    float envelope = 1.0f;
    if (t < ATTACK_TIME) {
      // アタックフェーズ：音量を徐々に上げる
      envelope = t / ATTACK_TIME;
    } else if (t > RELEASE_START) {
      // リリースフェーズ：音量を徐々に下げる
      envelope = (1.0f - t) / RELEASE_TIME;
    }
    // サステインフェーズ：最大音量を維持（envelope = 1.0）

    wave *= envelope;

    // 振幅スケール適用後、16bit signed intに変換
    float scaledWave = wave * AMPLITUDE_MAX * amplitudeScale;

    // クリッピング防止
    if (scaledWave > AMPLITUDE_MAX) scaledWave = AMPLITUDE_MAX;
    if (scaledWave < -AMPLITUDE_MAX) scaledWave = -AMPLITUDE_MAX;

    waveTable[i] = (int16_t)scaledWave;
  }
}

// ================================================================================
// オーディオ出力タスク
// ================================================================================
/**
 * オーディオ出力FreeRTOSタスク
 * 波形テーブルから音声データを読み出してI2S経由で出力
 * @param parameter タスクパラメータ（未使用）
 */
void audioTask(void *parameter) {
  size_t bytes_written;
  int16_t buffer[AUDIO_BUFFER_SIZE];

  // 位相アキュムレータ（固定小数点演算用）
  uint32_t phaseAccumulator = 0;
  // 位相インクリメント値（再生速度に応じて変化）
  uint32_t phaseIncrement = (uint32_t)((float)WAVE_TABLE_SIZE * playbackRate * 65536.0f / SAMPLE_RATE);

  while (true) {
    if (isPlaying) {
      // 再生速度の変更を反映
      phaseIncrement = (uint32_t)((float)WAVE_TABLE_SIZE * playbackRate * 65536.0f / SAMPLE_RATE);

      // バッファを埋める
      for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
        // 固定小数点から整数部を取り出し、波形テーブルのインデックスとする
        uint32_t tableIndex = (phaseAccumulator >> 16) % WAVE_TABLE_SIZE;
        buffer[i] = waveTable[tableIndex];

        // 位相を進める
        phaseAccumulator += phaseIncrement;
      }

      // I2S経由で音声データを出力
      i2s_write(I2S_NUM_0, buffer, sizeof(buffer), &bytes_written, portMAX_DELAY);
    } else {
      // 無音出力（再生停止中）
      memset(buffer, 0, sizeof(buffer));
      i2s_write(I2S_NUM_0, buffer, sizeof(buffer), &bytes_written, portMAX_DELAY);

      // 停止中は位相をリセット
      phaseAccumulator = 0;
    }

    // 他のタスクに処理を譲る
    vTaskDelay(1);
  }
}

// ボタン割り込み処理
void IRAM_ATTR buttonISR() {
  buttonPressed = true;
}

// WiFi設定とWebサーバーセットアップ
void setupWiFi() {
  // WiFi.begin("YourWiFiSSID", "YourWiFiPassword");
  WiFi.softAP("ESP32-VoiceSynth", "12345678");  // APモードの場合
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// Webサーバーのハンドラー
void handleRoot() {
  String html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>Voice Synthesizer Control</title>
    <meta charset='utf-8'>
    <style>
        body { font-family: Arial; margin: 20px; }
        .control { margin: 10px 0; }
        input[type='range'] { width: 300px; }
        button { padding: 10px 20px; margin: 5px; }
    </style>
</head>
<body>
    <h1>人工咽頭制御システム</h1>
    
    <div class='control'>
        <label>再生速度: <span id='rateValue'>1.0</span></label><br>
        <input type='range' id='playbackRate' min='0.5' max='2.0' step='0.1' value='1.0'>
    </div>
    
    <div class='control'>
        <label>音量: <span id='ampValue'>0.8</span></label><br>
        <input type='range' id='amplitude' min='0.1' max='1.0' step='0.1' value='0.8'>
    </div>
    
    <div class='control'>
        <button onclick='playSound()'>音声再生</button>
        <button onclick='stopSound()'>停止</button>
        <button onclick='updateWave()'>波形更新</button>
    </div>
    
    <div class='control'>
        <h3>ステータス</h3>
        <p id='status'>待機中</p>
    </div>

    <script>
        function updateValue(id, spanId) {
            document.getElementById(spanId).innerText = document.getElementById(id).value;
        }
        
        document.getElementById('playbackRate').oninput = function() {
            updateValue('playbackRate', 'rateValue');
        }
        
        document.getElementById('amplitude').oninput = function() {
            updateValue('amplitude', 'ampValue');
        }
        
        function playSound() {
            fetch('/play', {method: 'POST'});
            document.getElementById('status').innerText = '再生中';
        }
        
        function stopSound() {
            fetch('/stop', {method: 'POST'});
            document.getElementById('status').innerText = '停止';
        }
        
        function updateWave() {
            const rate = document.getElementById('playbackRate').value;
            const amp = document.getElementById('amplitude').value;
            
            fetch('/update', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rate: parseFloat(rate), amplitude: parseFloat(amp)})
            });
            document.getElementById('status').innerText = 'パラメータ更新';
        }
    </script>
</body>
</html>
)";
  server.send(200, "text/html", html);
}

void handlePlay() {
  isPlaying = true;
  server.send(200, "text/plain", "Playing");
  digitalWrite(LED_PIN, HIGH);
}

void handleStop() {
  isPlaying = false;
  server.send(200, "text/plain", "Stopped");
  digitalWrite(LED_PIN, LOW);
}

/**
 * パラメータ更新ハンドラー
 * PythonチューニングシステムまたはWebUIからのパラメータ更新を処理
 */
void handleUpdate() {
  if (!server.hasArg("plain")) {
    server.send(400, "text/plain", "Bad Request: No JSON body");
    Serial.println("Error: No JSON body in update request");
    return;
  }

  // JSONデータをパース
  DynamicJsonDocument doc(JSON_BUFFER_SIZE);
  DeserializationError error = deserializeJson(doc, server.arg("plain"));

  if (error) {
    server.send(400, "text/plain", "Bad Request: Invalid JSON");
    Serial.printf("Error: JSON parse failed: %s\n", error.c_str());
    return;
  }

  // パラメータ更新
  bool updated = false;

  if (doc.containsKey("rate")) {
    playbackRate = doc["rate"];
    updated = true;
  }

  if (doc.containsKey("amplitude")) {
    amplitudeScale = doc["amplitude"];
    updated = true;
  }

  // 倍音構成の更新
  if (doc.containsKey("harmonics")) {
    JsonArray harmonics = doc["harmonics"];
    int count = min((int)harmonics.size(), NUM_HARMONICS);
    for (int i = 0; i < count; i++) {
      harmonicWeights[i] = harmonics[i];
    }
    updated = true;
  }

  if (updated) {
    // 波形テーブルを再生成
    generateBaseWaveform();

    server.send(200, "text/plain", "Updated");
    Serial.println("Parameters updated:");
    Serial.printf("  Rate: %.3f\n", playbackRate);
    Serial.printf("  Amplitude: %.3f\n", amplitudeScale);
    Serial.print("  Harmonics: [");
    for (int i = 0; i < NUM_HARMONICS; i++) {
      Serial.printf("%.3f%s", harmonicWeights[i], i < NUM_HARMONICS - 1 ? ", " : "");
    }
    Serial.println("]");
  } else {
    server.send(400, "text/plain", "Bad Request: No valid parameters");
    Serial.println("Error: No valid parameters in update request");
  }
}

/**
 * セットアップ関数
 * システム初期化を行う
 */
void setup() {
  // シリアル通信初期化
  Serial.begin(115200);
  delay(100);
  Serial.println("\n========================================");
  Serial.println("人工咽頭システム起動中...");
  Serial.println("========================================");

  // GPIO設定
  Serial.println("[1/7] GPIO設定中...");
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);
  Serial.println("  GPIO設定完了");

  // SPIFFS初期化
  Serial.println("[2/7] SPIFFS初期化中...");
  if (!SPIFFS.begin(true)) {
    Serial.println("  警告: SPIFFS初期化失敗");
  } else {
    Serial.println("  SPIFFS初期化完了");
  }

  // I2S初期化
  Serial.println("[3/7] I2Sオーディオ初期化中...");
  if (!setupI2S()) {
    Serial.println("  エラー: I2S初期化失敗 - システムを停止します");
    while (1) {
      delay(1000);
    }
  }

  // 基本波形生成
  Serial.println("[4/7] 波形テーブル生成中...");
  generateBaseWaveform();
  Serial.println("  波形生成完了");

  // WiFi & Webサーバー設定
  Serial.println("[5/7] WiFi接続中...");
  setupWiFi();

  Serial.println("[6/7] Webサーバー起動中...");
  server.on("/", handleRoot);
  server.on("/play", HTTP_POST, handlePlay);
  server.on("/stop", HTTP_POST, handleStop);
  server.on("/update", HTTP_POST, handleUpdate);
  server.begin();
  Serial.println("  Webサーバー起動完了");

  // 音声出力タスク開始
  Serial.println("[7/7] オーディオタスク起動中...");
  xTaskCreatePinnedToCore(
    audioTask,           // タスク関数
    "AudioTask",         // タスク名
    TASK_STACK_SIZE,     // スタックサイズ
    NULL,                // パラメータ
    TASK_PRIORITY,       // 優先度
    NULL,                // タスクハンドル
    TASK_CORE            // 実行コア
  );
  Serial.println("  オーディオタスク起動完了");

  // 起動完了
  Serial.println("\n========================================");
  Serial.println("システム準備完了!");
  Serial.println("========================================");
  Serial.printf("WebUI: http://%s\n", WiFi.localIP().toString().c_str());
  Serial.printf("AP SSID: %s\n", WIFI_AP_SSID);
  Serial.println("========================================\n");
}

/**
 * メインループ
 * Webサーバーリクエスト処理とボタン入力処理
 */
void loop() {
  // Webサーバーのクライアントリクエストを処理
  server.handleClient();

  // ボタン割り込みフラグをチェック
  if (buttonPressed) {
    buttonPressed = false;

    // 再生状態をトグル
    isPlaying = !isPlaying;

    // LED状態を更新
    digitalWrite(LED_PIN, isPlaying ? HIGH : LOW);

    // シリアルログ出力
    if (isPlaying) {
      Serial.println("▶ 音声再生開始");
    } else {
      Serial.println("■ 音声停止");
    }
  }

  // CPU負荷軽減のための小休止
  delay(10);
}