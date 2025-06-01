#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>
#include <ArduinoJson.h>

// ピン定義
#define I2S_BCLK_PIN    42  // BCK pin
#define I2S_LRCK_PIN    2   // LRCK pin  
#define I2S_DATA_PIN    41  // DIN pin
#define BUTTON_PIN      0   // 発声トリガーボタン
#define LED_PIN         48  // 状態表示LED

// 音声パラメータ
#define SAMPLE_RATE     22050
#define WAVE_TABLE_SIZE 1024
#define AMPLITUDE_MAX   16383  // 16bit signed の半分

// グローバル変数
bool isPlaying = false;
bool buttonPressed = false;
float playbackRate = 1.0;
float amplitudeScale = 0.8;
int16_t waveTable[WAVE_TABLE_SIZE];
WebServer server(80);

// I2S設定
void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    // .dma_desc_num = 8,
    // .dma_frame_num = 64,
    .use_apll = false,
    .tx_desc_auto_clear = true
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCLK_PIN,
    .ws_io_num = I2S_LRCK_PIN,
    .data_out_num = I2S_DATA_PIN,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}

// 基本波形テーブルの生成（仮の「あー」音をシミュレート）
void generateBaseWaveform() {
  for (int i = 0; i < WAVE_TABLE_SIZE; i++) {
    float t = (float)i / WAVE_TABLE_SIZE;
    
    // 基本波（150Hz想定）+ 倍音構成で「あー」音を近似
    float wave = 0.0;
    wave += 0.8 * sin(2 * PI * t);        // 基本波
    wave += 0.4 * sin(2 * PI * t * 2);    // 2倍音
    wave += 0.2 * sin(2 * PI * t * 3);    // 3倍音
    wave += 0.1 * sin(2 * PI * t * 4);    // 4倍音
    wave += 0.05 * sin(2 * PI * t * 5);   // 5倍音
    
    // フォルマント風の特性を追加
    float formant = 0.3 * sin(2 * PI * t * 8) * exp(-t * 2);
    wave += formant;
    
    // エンベロープ（アタック・リリース）
    float envelope = 1.0;
    if (t < 0.1) {
      envelope = t / 0.1;  // アタック
    } else if (t > 0.8) {
      envelope = (1.0 - t) / 0.2;  // リリース
    }
    
    wave *= envelope;
    
    // 16bit signed に変換
    waveTable[i] = (int16_t)(wave * AMPLITUDE_MAX * amplitudeScale);
  }
}

// 音声出力タスク
void audioTask(void *parameter) {
  size_t bytes_written;
  int16_t buffer[128];
  uint32_t sampleIndex = 0;
  uint32_t phaseAccumulator = 0;
  uint32_t phaseIncrement = (uint32_t)((float)WAVE_TABLE_SIZE * playbackRate * 65536.0 / SAMPLE_RATE);
  
  while (true) {
    if (isPlaying) {
      // バッファを埋める
      for (int i = 0; i < 128; i++) {
        uint32_t tableIndex = phaseAccumulator >> 16;
        buffer[i] = waveTable[tableIndex % WAVE_TABLE_SIZE];
        phaseAccumulator += phaseIncrement;
      }
      
      // I2S出力
      i2s_write(I2S_NUM_0, buffer, sizeof(buffer), &bytes_written, portMAX_DELAY);
    } else {
      // 無音出力
      memset(buffer, 0, sizeof(buffer));
      i2s_write(I2S_NUM_0, buffer, sizeof(buffer), &bytes_written, portMAX_DELAY);
    }
    
    vTaskDelay(1); // 他のタスクに処理を譲る
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

void handleUpdate() {
  if (server.hasArg("plain")) {
    DynamicJsonDocument doc(1024);
    deserializeJson(doc, server.arg("plain"));
    
    playbackRate = doc["rate"];
    amplitudeScale = doc["amplitude"];
    
    // 波形テーブルを再生成
    generateBaseWaveform();
    
    server.send(200, "text/plain", "Updated");
    Serial.printf("Parameters updated: Rate=%.2f, Amplitude=%.2f\n", playbackRate, amplitudeScale);
  } else {
    server.send(400, "text/plain", "Bad Request");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("人工咽頭システム起動中...");
  
  // ピン設定
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);
  
  // SPIFFS初期化
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS initialization failed!");
  }
  
  // I2S初期化
  setupI2S();
  Serial.println("I2S initialized");
  
  // 基本波形生成
  generateBaseWaveform();
  Serial.println("Base waveform generated");
  
  // WiFi & Webサーバー設定
  setupWiFi();
  server.on("/", handleRoot);
  server.on("/play", HTTP_POST, handlePlay);
  server.on("/stop", HTTP_POST, handleStop);
  server.on("/update", HTTP_POST, handleUpdate);
  server.begin();
  Serial.println("Web server started");
  
  // 音声出力タスク開始
  xTaskCreatePinnedToCore(audioTask, "AudioTask", 4096, NULL, 1, NULL, 1);
  
  Serial.println("システム準備完了!");
  Serial.printf("WebUI: http://%s\n", WiFi.localIP().toString().c_str());
}

void loop() {
  server.handleClient();
  
  // ボタン処理
  if (buttonPressed) {
    buttonPressed = false;
    isPlaying = !isPlaying;
    digitalWrite(LED_PIN, isPlaying ? HIGH : LOW);
    Serial.println(isPlaying ? "音声再生開始" : "音声停止");
  }
  
  delay(10);
}