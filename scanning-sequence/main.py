import RPi.GPIO as gpio
import time

gpio.setmode(gpio.BOARD)

pins = {
    "n450_p1": 1,
    "n450_p2": 2,
    "j660_p1": 3,
    "j660_p2": 4,
    "g730_p1": 5,
    "g730_p2": 6,
    "k850_p1": 7,
    "k850_p2": 8,
    "j940_p1": 9,
    "h940_p2": 10
}

for k in pins.values():
    gpio.setup(k, gpio.OUT)

screen_input = False

if screen_input:
    gpio.output(pins["n450_p1"], gpio.HIGH)
    gpio.output(pins["n450_p2"], gpio.HIGH)
    time.sleep(.1)
    # camera.take_snapshot()
    time.sleep(.1)
    gpio.output(pins["n450_p1"], gpio.LOW)
    gpio.output(pins["n450_p2"], gpio.LOW)

    gpio.output(pins.get("j660_p1"), gpio.HIGH)
    gpio.output(pins.get("j660_p2"), gpio.HIGH)
    time.sleep(.1)
    # camera.take_snapshot()
    time.sleep(.1)
    gpio.output(pins.get("j660_p1"), gpio.LOW)
    gpio.output(pins.get("j660_p2"), gpio.LOW)

    gpio.output(pins.get("j660_p1"), gpio.HIGH)
    gpio.output(pins.get("j660_p2"), gpio.HIGH)
    time.sleep(.1)
    # camera.take_snapshot()
    time.sleep(.1)
    gpio.output(pins.get("j660_p1"), gpio.LOW)
    gpio.output(pins.get("j660_p2"), gpio.LOW)
    