import mraa

# This example will change the LCD backlight on the Grove-LCD RGB backlight
# to a nice shade of purple
x = mraa.I2c(0)
x.address(0x62)

# initialise device
x.writeReg(0, 0)
x.writeReg(1, 0)

# sent RGB color data
x.writeReg(0x08, 0xAA)
x.writeReg(0x04, 255)
x.writeReg(0x02, 255)
