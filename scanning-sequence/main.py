import RPi.GPIO as gpio
import time
from picamera2 import Picamera2

gpio.setmode(gpio.BCM)

pins = {
    "n450": 1,
    "j660": 2,
    "g730": 3,
    "k850": 4,
    "h940": 5
}

def init():
	for k in pins.values():
    	gpio.setup(k, gpio.OUT)

	cam = Picamera2()

def sequence():
    gpio.output(pins["n450"], gpio.HIGH)
    time.sleep(.1)
    cam.start_and_capture_file("test1.jpg")
    time.sleep(.1)
    gpio.output(pins["n450"], gpio.LOW)

    gpio.output(pins["j660"], gpio.HIGH)
    time.sleep(.1)
    # camera.take_snapshot()
    time.sleep(.1)
    gpio.output(pins["j660"], gpio.LOW)

    gpio.output(pins["g730"], gpio.HIGH)
    time.sleep(.1)
    # camera.take_snapshot()
    time.sleep(.1)
    gpio.output(pins["g730"], gpio.LOW)

	gpio.output(pins["k850"], gpio.HIGH)
	time.sleep(.1)
	# camera.take_snapshot()
	time.sleep(.1)
	gpio.output(pins["k850"], gpio.LOW)

	gpio.output(pins["h940"], gpio.HIGH)
	time.sleep(.1)
	# camera.take_snapshot()
	time.sleep(.1)
	gpio.output(pins["h940"], gpio.LOW)
