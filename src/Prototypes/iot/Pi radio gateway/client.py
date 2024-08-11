import serial
import paho.mqtt.client as mqtt

import Adafruit_BBIO.UART as UART

# UART configuration change as needed
uart_device = '/dev/ttyO1'
uart_baudrate = 9600

# MQTT configuration if configured on local machine
mqtt_broker = 'localhost'
mqtt_port = 1883
mqtt_topic = 'SIT764/projectEcho'

# Connect to the MQTT broker
client = mqtt.Client("Client")

client.username_pw_set("username", "password")

def on_connect(client, userdata, flags, rc):
    print('Connected to broker')

def on_publish(client, userdata, mid):
    print('Message published')

client.on_connect = on_connect
client.on_publish = on_publish

client.connect(mqtt_broker)
client.loop_start()

# Connect the UART device on UART1
UART.setup("UART1")

uart = serial.Serial(port=uart_device,baudrate=uart_baudrate)
uart.close()
uart.open()


try:
    while True:
        # Wait until you have some text waiting on the serial line
        # when you have a line, read it and publish it to MQTT
        # Then loop again
        if uart.in_waiting>0:

            uart_data = uart.readline()
            client.publish(mqtt_topic, uart_data)

except:
    print("Error")

finally:
    client.loop_stop()
    client.disconnect()
    uart.close()
