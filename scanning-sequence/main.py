import RPi.GPIO as gpio
import time
from picamera2 import Picamera2
import smbus2
from luma.core.interface.serial import spi
from luma.lcd.device import st7789
from luma.core.render import canvas

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
time.sleep(.2)

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


PIN_DC = 25
PIN_RST = 5
PIN_TP_INT = 6
PIN_TP_RST = 12

gpio.setup(PIN_DC, gpio.OUT)
gpio.setup(PIN_RST, gpio.OUT)
gpio.setup(PIN_TP_INT, gpio.IN, pull_up_down=gpio.PUD_UP)
gpio.setup(PIN_TP_RST, gpio.OUT)

# reset touch controller
gpio.output(PIN_TP_RST, gpio.LOW)
time.sleep(0.05)
gpio.output(PIN_TP_RST, gpio.HIGH)
time.sleep(0.05)

# SPI for LCD
serial = spi(
    port=0,
    device=0,
    gpio_DC=PIN_DC,
    gpio_RST=PIN_RST,
    bus_speed_hz=40_000_000
)

device = st7789(serial, width=240, height=240, rotate=0)

# i2c bus for touch (check addr with i2cdetect)
I2C_BUS = 1
TP_ADDR = 0x38   # change if i2cdetect shows something else
bus = smbus2.SMBus(I2C_BUS)

BTN_RECT = (40, 80, 200, 160)


def draw_screen(pressed=False):
    with canvas(device) as draw:
        # background
        draw.rectangle((0, 0, 239, 239), outline="black", fill="black")

        # button color
        fill = "grey" if not pressed else "white"
        outline = "white"

        draw.rectangle(BTN_RECT, outline=outline, fill=fill)

        # label text
        text = "START SCAN"
        w, h = draw.textsize(text)
        bx0, by0, bx1, by1 = BTN_RECT
        tx = bx0 + (bx1 - bx0 - w) // 2
        ty = by0 + (by1 - by0 - h) // 2
        draw.text((tx, ty), text, fill="black" if pressed else "black")


def on_button_press():
    sequence()
    # add debug if smth wrong


def read_touch():
    try:
        data = bus.read_i2c_block_data(TP_ADDR, 0x02, 6)
        n_points = data[0] & 0x0F
        if n_points == 0:
            return None, None

        x = ((data[1] & 0x0F) << 8) | data[2]
        y = ((data[3] & 0x0F) << 8) | data[4]

        return x, y
    except Exception as e:
        # print("touch read error:", e)
        return None, None


def in_button(x, y):
    x0, y0, x1, y1 = BTN_RECT
    return x0 <= x <= x1 and y0 <= y <= y1


def main():
    try:
        draw_screen(pressed=False)

        while True:
            x, y = read_touch()
            if x is not None and y is not None:

                if in_button(x, y):
                    draw_screen(pressed=True)
                    on_button_press()
                    time.sleep(0.3)
                    draw_screen(pressed=False)

            time.sleep(0.02)
    finally:
        cam.stop()
        gpio.cleanup()


if __name__ == "__main__":
    main()
