from GPIOLibrary import GPIOProcessor
import time

i = 0
ID = ""
while i < 6:



    GP = GPIOProcessor()

    try:
        Pin29 = GP.getPin29()
        Pin29.out()

        Pin30 = GP.getPin30()
        Pin30.input()

    finally:
        GP.cleanup()
    i += 1
