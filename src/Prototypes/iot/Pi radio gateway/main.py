

from utime import sleep_ms,ticks_ms
from machine import Pin, UART
from sx1262 import SX1262

# Allow flashing LED for user awareness of incoming messages
LED = Pin(25, Pin.OUT)

# UART configuration change as needed
# Must be the same at the other end
uart_baudrate = 9600

#Config the UART bus
uart = UART(0, baudrate=uart_baudrate, tx= Pin(0), rx=Pin(1),bits=8, parity=None, stop=1)

# Config the SPI for LOra connection
sx = SX1262(spi_bus=1, clk=10, mosi=11, miso=12, cs=3, irq=20, rst=15, gpio=2)

# LORA settings DO NOT CHANGE if using inside Australia
sx.begin(freq=915, bw=125.0, sf=12, cr=8, syncWord=0x12,
         power=-5, currentLimit=60.0, preambleLength=8,
         implicit=False, implicitLen=0xFF,
         crcOn=True, txIq=False, rxIq=False,
         tcxoVoltage=1.7, useRegulatorLDO=False, blocking=True)


# LoRa receive message callback
def receive_callback(events):
    if events & SX1262.RX_DONE:
        msg, err = sx.recv()
        if msg != None:

            # Flash LED if message is received.
            LED.value(1)
            sleep_ms(1000)
            LED.value(0)

            error = SX1262.STATUS[err]
            print(msg.decode("utf-8"))

            # Send contents of received message to UART
            send_UART(msg.decode("utf-8"))
        
      
# Send UART to Pi
def send_UART(msg):
    # Ensure to end line with \n to end the line
    uart.write(f"{msg}\n")
    
# Configure the receive LoRa callback
sx.setBlockingCallback(False, receive_callback)

now = ticks_ms()

while True:
    
    if ticks_ms() - now >= 2000:
       now = ticks_ms()
       
       # We have to stay doing something to ensure program still runs
       print("waiting for messages")
        


