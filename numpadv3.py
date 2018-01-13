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

# initialize gpio 24
gpio_2 = mraa.Gpio(30)

gpio_3 = mraa.Gpio(26)
gpio_4 = mraa.Gpio(24)

# set gpio 23 to output
gpio_1.dir(mraa.DIR_IN)

# set gpio 24 to output
gpio_2.dir(mraa.DIR_IN)

gpio_3.dir(mraa.DIR_IN)
gpio_4.dir(mraa.DIR_IN)

i = 0
ID = ""

while i < 4:
    i += 1
    if gpio_1 & gpio_2 == 1:
        ID += "1"
        print (ID)
    elif gpio_1 & gpio_3 == 1:
        ID += "2"
        print(ID)
    elif gpio_1 & gpio_4 == 1:
        ID += "3"
        print(ID)
    elif i == 3:
        i = 0
        ID = ""


