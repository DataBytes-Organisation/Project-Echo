/* ESP32 HTTP IoT Server Example for Wokwi.com

  https://wokwi.com/projects/320964045035274834

  To test, you need the Wokwi IoT Gateway, as explained here:

  https://docs.wokwi.com/guides/esp32-wifi#the-private-gateway

  Then start the simulation, and open http://localhost:9080
  in another browser tab.

  Note that the IoT Gateway requires a Wokwi Club subscription.
  To purchase a Wokwi Club subscription, go to https://wokwi.com/club
*/

#include <WiFi.h>
#include <WiFiClient.h>
#include <WebServer.h>
#include <uri/UriBraces.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#define WIFI_SSID "Wokwi-GUEST"
#define WIFI_PASSWORD ""
// Defining the WiFi channel speeds up the connection:
#define WIFI_CHANNEL 6

WebServer server(80);

const int LED1 = 26;
const int LED2 = 27;

bool led1State = false;
bool led2State = false;

const char* node_id = "node_3_2";
const char* server_ip = "192.168.1.9";

#define DHTPIN 15     // Digital pin connected to the DHT sensor
#define DHTTYPE DHT22 // DHT22 (AM2302)
DHT dht(DHTPIN, DHTTYPE);
Adafruit_MPU6050 mpu;
void sendHtml() {
  String response = R"(
    <!DOCTYPE html><html>
      <head>
        <title>ESP32 Web Server Demo</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          html { font-family: sans-serif; text-align: center; }
          body { display: inline-flex; flex-direction: column; }
          h1 { margin-bottom: 1.2em; } 
          h2 { margin: 0; }
          div { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto; grid-auto-flow: column; grid-gap: 1em; }
          .btn { background-color: #5B5; border: none; color: #fff; padding: 0.5em 1em;
                 font-size: 2em; text-decoration: none }
          .btn.OFF { background-color: #333; }
        </style>
      </head>
            
      <body>
        <h1>ESP32 Web Server</h1>

        <div>
          <h2>LED 1</h2>
          <a href="/toggle/1" class="btn LED1_TEXT">LED1_TEXT</a>
          <h2>LED 2</h2>
          <a href="/toggle/2" class="btn LED2_TEXT">LED2_TEXT</a>
        </div>
      </body>
    </html>
  )";
  response.replace("LED1_TEXT", led1State ? "ON" : "OFF");
  response.replace("LED2_TEXT", led2State ? "ON" : "OFF");
  server.send(200, "text/html", response);
}

void setup(void) {
  Serial.begin(115200);
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  dht.begin();

  // Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD, WIFI_CHANNEL);
  Serial.print("Connecting to WiFi ");
  Serial.print(WIFI_SSID);
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    Serial.print(".");
  }
  Serial.println(" Connected!");

  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Register the node
  HTTPClient http;
  String url = String("http://" + String(server_ip) + ":9000/iot/nodes/" + String(node_id) + "/register");
  http.begin(url);
  int httpResponseCode = http.PUT("");
  
  if (httpResponseCode > 0) {
    Serial.print("Node registration successful, response code: ");
    Serial.println(httpResponseCode);
  } else {
    Serial.print("Node registration failed, error: ");
    Serial.println(http.errorToString(httpResponseCode));
  }
  http.end();

  server.on("/", sendHtml);

  server.on(UriBraces("/toggle/{}"), []() {
    String led = server.pathArg(0);
    Serial.print("Toggle LED #");
    Serial.println(led);

    switch (led.toInt()) {
      case 1:
        led1State = !led1State;
        digitalWrite(LED1, led1State);
        break;
      case 2:
        led2State = !led2State;
        digitalWrite(LED2, led2State);
        break;
    }

    sendHtml();
  });

  server.begin();
  Serial.println("HTTP server started");
}

unsigned long lastHeartbeat = 0;
unsigned long lastSensorUpdate = 0;
const unsigned long HEARTBEAT_INTERVAL = 10000; // 10 seconds in milliseconds
const unsigned long SENSOR_UPDATE_INTERVAL = 15000; // 15 seconds in milliseconds

void sendHeartbeat() {
  Serial.println("Sending heartbeat...");
  HTTPClient http;
  String url = String("http://") + String(server_ip) + ":9000/iot/nodes/" + String(node_id) + "/heartbeat";
  http.begin(url);
  
  // Set content type header
  http.addHeader("Content-Type", "application/json");
  
  // Create JSON payload
  String payload = "{\"message\": \"ESP32 is alive\"}";
  
  int httpResponseCode = http.PUT(payload);
  
  if (httpResponseCode > 0) {
    Serial.print("Heartbeat sent successfully, response code: ");
    Serial.println(httpResponseCode);
  } else {
    Serial.print("Error sending heartbeat: ");
    Serial.println(http.errorToString(httpResponseCode));
  }
  
  http.end();
}

void sendSensorUpdates() {
  // Read DHT sensor
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  // Check if DHT reads failed
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  // Read MPU6050 sensor
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Prepare the updates array
  String updates = String("[") +
    // DHT22 update
    "{\"component_id\": \"dht11\", \"data\": {" +
    "\"temperature\": " + String(temperature) + "," +
    "\"humidity\": " + String(humidity) +
    "}}," +
    // MPU6050 update
    "{\"component_id\": \"imu1\", \"data\": {" +
    "\"accelerometer\": {" +
    "\"x\": " + String(a.acceleration.x) + "," +
    "\"y\": " + String(a.acceleration.y) + "," +
    "\"z\": " + String(a.acceleration.z) +
    "}," +
    "\"gyroscope\": {" +
    "\"x\": " + String(g.gyro.x) + "," +
    "\"y\": " + String(g.gyro.y) + "," +
    "\"z\": " + String(g.gyro.z) +
    "}" +
    "}}" +
    "]";  

  // Send to server
  HTTPClient http;
  String url = String("http://") + String(server_ip) + ":9000/iot/nodes/" + String(node_id) + "/updates";
  http.begin(url);
  
  // Set content type header
  http.addHeader("Content-Type", "application/json");
  
  int httpResponseCode = http.POST(updates);
  
  if (httpResponseCode > 0) {
    Serial.println("Sensor updates sent successfully, response code: " + String(httpResponseCode));
  } else {
    Serial.println("Error sending sensor updates: " + http.errorToString(httpResponseCode));
  }
  
  http.end();
}

void loop(void) {
  server.handleClient();
  
  unsigned long currentMillis = millis();
  
  // Check for heartbeat
  if (lastHeartbeat == 0 || (currentMillis - lastHeartbeat) >= HEARTBEAT_INTERVAL) {
    sendHeartbeat();
    lastHeartbeat = currentMillis;
  }

  // Check for sensor updates
  if (lastSensorUpdate == 0 || (currentMillis - lastSensorUpdate) >= SENSOR_UPDATE_INTERVAL) {
    sendSensorUpdates();
    lastSensorUpdate = currentMillis;
  }
  
  delay(2);
}
