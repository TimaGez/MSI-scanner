import RPi.GPIO as gpio
import time
from picamera2 import Picamera2

gpio.setmode(gpio.BCM)

pins = {
	"n450": 17,
	"j660": 27,
	"g730": 22,
	"k850": 23,
	"h940": 24
}

cam = Picamera2()
cam.configure(cam.create_still_configuration())
cam.start()

for k in pins.values():
	gpio.setup(k, gpio.OUT)


def sequence():
	gpio.output(pins["n450"], gpio.HIGH)
	time.sleep(.1)
	cam.capture_file("test1.jpg")
	time.sleep(.1)
	gpio.output(pins["n450"], gpio.LOW)

	gpio.output(pins["j660"], gpio.HIGH)
	time.sleep(.1)
	cam.capture_file("test2.jpg")
	time.sleep(.1)
	gpio.output(pins["j660"], gpio.LOW)

	gpio.output(pins["g730"], gpio.HIGH)
	time.sleep(.1)
	cam.capture_file("test3.jpg")
	time.sleep(.1)
	gpio.output(pins["g730"], gpio.LOW)

	gpio.output(pins["k850"], gpio.HIGH)
	time.sleep(.1)
	cam.capture_file("test4.jpg")
	time.sleep(.1)
	gpio.output(pins["k850"], gpio.LOW)
	
	gpio.output(pins["h940"], gpio.HIGH)
	time.sleep(.1)
	cam.capture_file("test5.jpg")
	time.sleep(.1)
	gpio.output(pins["h940"], gpio.LOW)

sequence()