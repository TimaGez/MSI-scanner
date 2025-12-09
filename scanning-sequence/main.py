import RPi.GPIO as gpio
import time
from picamera2 import Picamera2
from datetime import date
from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import i2c
from luma.lcd.device import st7567

gpio.setmode(gpio.BCM)

pins = {
	"n450": 17,
	"j660": 27,
	"g730": 22,
	"k850": 23,
	"h940": 24
}

button_pin = 26

is_button_pressed = False

cam = Picamera2()
cam.configure(cam.create_still_configuration())
cam.start()
time.sleep(.2)

for k in pins.values():
	gpio.setup(k, gpio.OUT)
gpio.setup(button_pin, gpio.IN, pull_up_down = gpio.PUD_UP)

serial = i2c(port=1, address=0x3C)

device = st7567(serial, width=128, height=64)

WIDTH = device.width
HEIGHT = device.height
font = ImageFont.load_default()

last_state = gpio.input(button_pin)

counter = 0
day = date.today()

def sequence():
	global counter
	global day
	counter += 1
	gpio.output(pins["n450"], gpio.HIGH)
	time.sleep(.1)
	cam.capture_file(f"{day}-{counter}-450nm.jpg")
	time.sleep(.1)
	gpio.output(pins["n450"], gpio.LOW)

	gpio.output(pins["j660"], gpio.HIGH)
	time.sleep(.1)
	cam.capture_file(f"{day}-{counter}-660nm.jpg")
	time.sleep(.1)
	gpio.output(pins["j660"], gpio.LOW)

	gpio.output(pins["g730"], gpio.HIGH)
	time.sleep(.1)
	cam.capture_file(f"{day}-{counter}-730nm.jpg")
	time.sleep(.1)
	gpio.output(pins["g730"], gpio.LOW)

	gpio.output(pins["k850"], gpio.HIGH)
	time.sleep(.1)
	cam.capture_file(f"{day}-{counter}-850nm.jpg")
	time.sleep(.1)
	gpio.output(pins["k850"], gpio.LOW)
	
	gpio.output(pins["h940"], gpio.HIGH)
	time.sleep(.1)
	cam.capture_file(f"{day}-{counter}-940nm.jpg")
	time.sleep(.1)
	gpio.output(pins["h940"], gpio.LOW)

def setup_screen():
	global button_pin, last_state, WIDTH, HEIGHT, device, font
	try:
		while True:
			state = gpio.input(button_pin)
			
			last_state = state

			image = Image.new("1", (WIDTH, HEIGHT)) 
			draw = ImageDraw.Draw(image)
			draw.text((0,0), "TEST". font=font, fill=255)

			text = "PRESSED" if state == gpio.low else "released"
			draw.text((0,32), f"Button: {text}", font=font, fill=255)

			device.display(image)

			time.sleep(.02)
	except KeyboardInterrupt:
		pass
	finally:
		image = Image.new("1", (WIDTH, HEIGHT))
		device.display(image)
		gpio.cleanup()

def set_button_press(is_pressed):
	global is_button_pressed
	is_button_pressed = is_pressed

def button_pressed():
	global is_button_pressed
	return is_button_pressed

def on_button_press():
    sequence()
    # add debug if smth wrong

def main():
        while True:

            if button_pressed():
                on_button_press()

        cam.stop()
        gpio.cleanup()



if __name__ == "__main__":
    main()
