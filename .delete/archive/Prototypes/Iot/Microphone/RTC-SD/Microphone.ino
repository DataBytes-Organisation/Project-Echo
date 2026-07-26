//Credit to Great Scott for spybug project without which this project would not have been created
//https://www.youtube.com/user/greatscottlab (channel)
//https://youtu.be/7Hn4UFi9wvs (spybug_project)

/*
1. Edit pcmConfig.h
    a: On Uno or non-mega boards, #define buffSize 128. May need to increase.
    b: Uncomment #define ENABLE_RECORDING and #define BLOCK_COUNT 10000UL

2. See https://github.com/TMRh20/TMRpcm/wiki/Advanced-Features#wiki-recording-audio for
   additional informaiton.
*/

//adds necessary libraries
#include <SD.h>
#include <SPI.h>
#include <TMRpcm.h>
#include <Wire.h>
#include <RTClib.h>
#include <time.h>
//creates global variables
#define SD_ChipSelectPin 10
#define ECHO_TO_SERIAL   1 // echo data to serial port.

TMRpcm audio;

int file_number = 0;
volatile DateTime now;
volatile int count=0;
volatile bool recording_now = false;
const int record_time = 1000;
const int button_pin = 2;
const int recording_led_pin = 3;
const int mic_pin = A0;
const int sample_rate = 16000;
const int outpin = 4;
RTC_DS1307 RTC; // define the Real Time Clock object

#define ECHO_TO_SERIAL   1 // echo data to serial port. 


void initRTC()
{
  Wire.begin();
  if (!RTC.begin()) {
Serial.println("RTC failed");


  }
}


//code below executed each time the button is pressed down
void button_pushed() {
  
  //combines the neccessary info to find the get a char array of the file name
  char file_name[20];
  itoa(file_number,file_name,10);
 
  //strftime(file_name, sizeof(file_name),"%X",now.unixtime());
  strcat(file_name,".wav");

  if (!recording_now) {
    //isn't recording so starts recording & turns LED on
    recording_now = true;
    digitalWrite(recording_led_pin, HIGH);
    audio.startRecording(file_name, sample_rate, mic_pin);
    Serial.println(file_name);
  }
  else {
    //is recording so stops recording & turns LED off
    recording_now = false;
    digitalWrite(recording_led_pin, LOW);
    audio.stopRecording(file_name);
    file_number++;
  }
}

  ISR(TIMER2_COMPA_vect)
{
  
  if(count == record_time)
  {
    count = 0;
    digitalWrite(outpin, HIGH);
    delay(10);
    digitalWrite(outpin,LOW);
  }
  else if(!recording_now)
  {
    
    digitalWrite(outpin, HIGH);
    delay(1000);
    digitalWrite(outpin,LOW);
  }
  
  
  count ++;
  OCR2A = 0x9C;
}

void setup() {
  //initialises the serial connection between the arduino and any connected serial device
  Serial.begin(9600);
  Serial.println("loading...");
  cli();// stop interupts


  //set timer2 interrupt at 10Hz -- 8 Bit comparator counter
  // using timer 2 as the TMRpcm.h use timer 1 
  TCCR2A = 0;// set entire TCCR1A register to 0
  TCCR2B = 0;// same for TCCR1B
  TCNT2  = 0;//initialize counter value to 0
  // set compare match register for 1hz increments
  OCR2A = 156;// = (16*10^6) / (1*1024) - 1 (must be <256)
  // turn on CTC mode
  TCCR2B |= (1 << WGM12);
  // Set CS20 -> CS22 bits for 1024 prescaler
  TCCR2B |= (1 << CS20) | (1 << CS21)|(1 << CS22);  
  // enable timer compare interrupt
  TIMSK2 |= (1 << OCIE1A);

sei(); // allow interupts

  /**
   * connect to RTC
     Now we kick off the RTC by initializing the Wire library and poking the RTC to see if its alive.
  */
initRTC();

  //Sets up the pins
  pinMode(mic_pin, INPUT);
  pinMode(recording_led_pin, OUTPUT);
  pinMode(button_pin, INPUT_PULLUP);
  pinMode(outpin, OUTPUT);
  //Sets up the audio recording functionality
  attachInterrupt(digitalPinToInterrupt(button_pin), button_pushed, FALLING);
  SD.begin(SD_ChipSelectPin);
  audio.CSPin = SD_ChipSelectPin;
}




void loop() {

  //doesn't loop through any code. As we have attached an interupt the arduino will notify our code when the button_pin is FALLING(when pressed down)
}