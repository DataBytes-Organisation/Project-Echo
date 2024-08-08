/*
  Code has been adapted from LoRa Simple Gateway/Node Exemple
  created 05 August 2018
  by Luiz H. Cassettari

  Changes made to experiment with Range for Project Echo T2 2024
  This is the code for the NODE  
  
*/

#include <SPI.h>     
#include <LoRa.h>

const long frequency = 915E6;  // Australian specific Lora freq DO NOT CHANGE THIS

// Pin settings for the Lora shield.  If the Arduino board changes, this might need to be changed.
const int csPin = 10;
const int resetPin = 9;
const int irqPin = 2;

void setup() {

  // Set baud to 9600
  Serial.begin(9600);

  // Wait for serial to come up
  while (!Serial);

  // Configure the Lora shield connection as per pin config.
  LoRa.setPins(csPin, resetPin, irqPin);

  // Check to see if Lora is available.
  if (!LoRa.begin(frequency)) {
    Serial.println("Check Lora shield and correct Pin layout - Connection has failed.");
    while (true);
  }

  Serial.println("LoRa config succeeded.");

  // Callbacks for RX and TX
  LoRa.onReceive(onReceive);
  LoRa.onTxDone(onTxDone);

  // Set into RX mode while not TXing
  LoRa_rxMode();
}

void loop() {

  // Repeat send a message every 10 secs

  if (runEvery(10000)) { 

    String message = "Project ECHO Data RF test";

    LoRa_sendMessage(message); 

    Serial.println("Message TX: "+message);
  }
}

void LoRa_rxMode(){
  LoRa.enableInvertIQ();                // active invert I and Q signals
  LoRa.receive();                       // set receive mode
}

void LoRa_txMode(){
  LoRa.idle();                          // set standby mode
  LoRa.disableInvertIQ();               // normal mode
}

void LoRa_sendMessage(String message) {
  LoRa_txMode();                        // set tx mode
  LoRa.beginPacket();                   // start packet
  LoRa.print(message);                  // add payload
  LoRa.endPacket(true);                 // finish packet and send it
}

void onReceive(int packetSize) {
  String message = "";

  while (LoRa.available()) {
    message += (char)LoRa.read();
  }

  Serial.print("Node Receive: ");
  Serial.println(message);
}

void onTxDone() {
  LoRa_rxMode();
}

boolean runEvery(unsigned long interval)
{
  static unsigned long previousMillis = 0;
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval)
  {
    previousMillis = currentMillis;
    return true;
  }
  return false;
}