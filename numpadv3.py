#!/usr/bin/env python

# gpio example in python using mraa
#
# Author: Manivannan Sadhasivam <manivannan.sadhasivam@linaro.org>
#
# Usage: Toggles GPIO 23 and 24
#
# Execution: sudo python mraa_gpio.py

import mraa
import time

# initialize gpio 23
gpio_1 = mraa.Gpio(29)
gpio_2 = mraa.Gpio(32)
gpio_3 = mraa.Gpio(30)
gpio_4 = mraa.Gpio(26)

# set GPIO to input
gpio_1.dir(mraa.DIR_IN)
gpio_1.write(1)
gpio_2.dir(mraa.DIR_IN)
gpio_2.write(1)
gpio_3.dir(mraa.DIR_IN)
gpio_3.write(1)
gpio_4.dir(mraa.DIR_IN)
gpio_4.write(1)

i = 0
ID = ""

while True:
    i += 1
    time.sleep(1)
    print("GPIO29 ", int(gpio_1.read()))
    print("GPIO32 ", int(gpio_2.read()))
    print("GPIO30 ", int(gpio_3.read()))
    print("GPIO26 ", int(gpio_4.read()))
    if (int(gpio_1.read()) and int(gpio_2.read())) == 1:
        ID += "1"
        print (ID)
    elif (int(gpio_1.read()) and int(gpio_3.read())) == 1:
        ID += "2"
        print(ID)
    elif (int(gpio_1.read()) and int(gpio_4.read())) == 1:
        ID += "3"
        print(ID)
    elif len(ID) == 3:
        print(ID)
        i = 0
        ID = ""



