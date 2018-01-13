import mraa
import time
# Refer to the pin-out diagram for the GPIO number to silk print mapping
# in this example the number 2 maps to P10 on LinkIt Smart 7688 board
pin = mraa.Gpio(24)
pin.dir(mraa.DIR_IN)
while True:
    print ("P10 state:", pin.read())
    time.sleep(0.3)