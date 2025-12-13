import RPi.GPIO as gpio
import time
from picamera2 import Picamera2
from datetime import date
from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

gpio.setmode(gpio.BCM)

pins = {
    "n450": 17,
    "j660": 27,
    "g730": 22,
    "k850": 23,
    "h940": 24
}

button_pin = 26

for k in pins.values():
    gpio.setup(k, gpio.OUT)

gpio.setup(button_pin, gpio.IN, pull_up_down=gpio.PUD_UP)

cam = Picamera2()
cam.configure(cam.create_still_configuration())
cam.start()
time.sleep(0.2)

serial = i2c(port=1, address=0x3C)
device = ssd1306(serial, width=128, height=64)
device.contrast(255)

WIDTH = device.width
HEIGHT = device.height
font = ImageFont.load_default()

counter = 0
day = date.today()
last_state = gpio.input(button_pin)


def sequence():
    global counter, day

    counter += 1

    # 450 nm
    gpio.output(pins["n450"], gpio.HIGH)
    time.sleep(0.1)
    cam.capture_file(f"{day}-{counter}-450nm.jpg")
    time.sleep(0.1)
    gpio.output(pins["n450"], gpio.LOW)

    # 660 nm
    gpio.output(pins["j660"], gpio.HIGH)
    time.sleep(0.1)
    cam.capture_file(f"{day}-{counter}-660nm.jpg")
    time.sleep(0.1)
    gpio.output(pins["j660"], gpio.LOW)

    # 730 nm
    gpio.output(pins["g730"], gpio.HIGH)
    time.sleep(0.1)
    cam.capture_file(f"{day}-{counter}-730nm.jpg")
    time.sleep(0.1)
    gpio.output(pins["g730"], gpio.LOW)

    # 850 nm
    gpio.output(pins["k850"], gpio.HIGH)
    time.sleep(0.1)
    cam.capture_file(f"{day}-{counter}-850nm.jpg")
    time.sleep(0.1)
    gpio.output(pins["k850"], gpio.LOW)

    # 940 nm
    gpio.output(pins["h940"], gpio.HIGH)
    time.sleep(0.1)
    cam.capture_file(f"{day}-{counter}-940nm.jpg")
    time.sleep(0.1)
    gpio.output(pins["h940"], gpio.LOW)


def main():
    global last_state, counter

    try:
        while True:
            state = gpio.input(button_pin)

            if state == gpio.LOW and last_state == gpio.HIGH:
                sequence()

            last_state = state

            image = Image.new("1", (WIDTH, HEIGHT))
            draw = ImageDraw.Draw(image)

            draw.text((0, 0), "MSI Scanner", font=font, fill=255)
            draw.text((0, 16), f"Scan #: {counter}", font=font, fill=255)

            text = "PRESSED" if state == gpio.LOW else "released"
            draw.text((0, 32), f"Button: {text}", font=font, fill=255)

            device.display(image)

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        image = Image.new("1", (WIDTH, HEIGHT))
        device.display(image)
        cam.stop()
        gpio.cleanup()


if __name__ == "__main__":
    main()
