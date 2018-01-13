import mraa
import time
# Refer to the pin-out diagram for the GPIO number to silk print mapping
# in this example the number 2 maps to P10 on LinkIt Smart 7688 board
#from mraa import Result as r

rowVal = 0
colVal = 0
value = ""
pin1 = mraa.Gpio(24)
pin2 = mraa.Gpio(26)
pin3 = mraa.Gpio(30)
pin4 = mraa.Gpio(32)
pin5 = mraa.Gpio(34)
pin6 = mraa.Gpio(23)
pin7 = mraa.Gpio(25)
pin8 = mraa.Gpio(27)

pin1.dir(mraa.DIR_IN)
pin2.dir(mraa.DIR_IN)
pin3.dir(mraa.DIR_IN)
pin4.dir(mraa.DIR_IN)
pin5.dir(mraa.DIR_OUT)
pin6.dir(mraa.DIR_OUT)
pin7.dir(mraa.DIR_OUT)
pin8.dir(mraa.DIR_OUT)

if(pin1 == None):
    print('Error')
#ret = pin1.dir(mraa.DIR_IN)

#if(ret!=mraa.SUCCESS):
#    print('s')

#pin8.write(1)
#pin1.write(1)
pin8.write(1)
pin7.write(1)
pin6.write(1)
pin5.write(1)
pin4.write(0)
pin3.write(0)
pin2.write(0)
pin1.write(0)

result = ""
while True:
    if(pin1.read() == 1):
        rowVal=1
    if(pin2.read() == 1):
        rowVal=2
    if(pin3.read() == 1):
        rowVal=3
    if(pin4.read() == 1):
        rowVal=4
    if not rowVal == 0:
        #num = pin.read()
        pin5.dir(mraa.DIR_IN)
        pin6.dir(mraa.DIR_IN)
        pin7.dir(mraa.DIR_IN)
        pin8.dir(mraa.DIR_IN)
        pin5.write(0)
        pin6.write(0)
        pin7.write(0)
        pin8.write(0)
        if rowVal==1:
            pin1.dir(mraa.DIR_OUT)
    if(pin3.read() == 1):
        rowVal=3
    if(pin4.read() == 1):
        rowVal=4
    if not rowVal == 0:
        #num = pin.read()
        pin5.dir(mraa.DIR_IN)
        pin6.dir(mraa.DIR_IN)
        pin7.dir(mraa.DIR_IN)
        pin8.dir(mraa.DIR_IN)
        pin5.write(0)
        pin6.write(0)
        pin7.write(0)
        pin8.write(0)
        if rowVal==1:
            pin1.dir(mraa.DIR_OUT)
            pin1.write(1)
        if rowVal==2:
            pin2.dir(mraa.DIR_OUT)
            pin2.write(1)
        if rowVal==3:
            pin3.dir(mraa.DIR_OUT)
            pin3.write(1)
        if rowVal==4:
            pin4.dir(mraa.DIR_OUT)
            pin4.write(1)
        if(pin5.read()==1):
            colVal = 4
        if(pin6.read()==1):
            colVal = 3
        if(pin7.read()==1):
            colVal = 2
        if(pin8.read()==1):
            colVal = 1
        #print (str(rowVal) + "   " +str(colVal))

    pin1.dir(mraa.DIR_IN)
    pin2.dir(mraa.DIR_IN)
    pin3.dir(mraa.DIR_IN)
    pin4.dir(mraa.DIR_IN)
    pin5.dir(mraa.DIR_OUT)
    pin6.dir(mraa.DIR_OUT)
    pin7.dir(mraa.DIR_OUT)
    pin8.dir(mraa.DIR_OUT)
    pin8.write(1)
    pin7.write(1)
    pin6.write(1)
    pin5.write(1)
    pin4.write(0)
    pin3.write(0)
    pin2.write(0)
    pin1.write(0)

    if(rowVal==1 and colVal ==1):
        value = "A"
    if(rowVal==2 and colVal ==1):
        value = "3"
    if(rowVal==3 and colVal ==1):
        value = "2"
    if(rowVal==4 and colVal ==1):
        value = "1"
    if(rowVal==1 and colVal ==2):
        value = "B"
    if(rowVal==2 and colVal ==2):
        value = "6"
    if(rowVal==3 and colVal ==2):
        value = "5"
    if(rowVal==4 and colVal ==2):
        value = "4"
    if(rowVal==1 and colVal ==3):
        value = "C"
    if(rowVal==2 and colVal ==3):
        value = "9"
    if(rowVal==3 and colVal ==3):
        value = "8"
    if(rowVal==4 and colVal ==3):
        value = "7"
    if(rowVal==1 and colVal ==4):
        value = "D"
    if(rowVal==2 and colVal ==4):
        value = "#"
    if(rowVal==3 and colVal ==4):
        value = "0"
    if(rowVal==4 and colVal ==4):
        value = "*"
        break

    if not value == "":
        print(value)
    result+=value
    time.sleep(0.3)
    colVal=0
    rowVal=0
    value=""

print(result)


