import numpy as np
import os
import RPi.GPIO as gpio
import time
from picamera2 import Picamera2
from datetime import date, datetime
from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

gpio.setmode(gpio.BCM)

pins = {
    "n450": 17,
    "g730": 22,
    "k850": 23,
}

button_pin = 26

for k in pins.values():
    gpio.setup(k, gpio.OUT)

gpio.setup(button_pin, gpio.IN, pull_up_down=gpio.PUD_UP)

cam = Picamera2()
cam.configure(cam.create_still_configuration(
    main={"format":"YUV420"}))
cam.start()

cam.set_controls({
    "AfMode": 0,
    "LensPosition": 4.5
})

time.sleep(1.0)
cam.set_controls({
    "AeEnable": False,
    "AwbEnable": False,
    "ColourGains": (1.0, 1.0),
    "Saturation": 1.0,
    "Sharpness": 0.0,
    "NoiseReductionMode": 0
})

CAPTURE_SETTINGS = {
    "still": {"ExposureTime": 350, "AnalogueGain": 1.0, "LensPosition": 4.5},

    "450":  {"ExposureTime": 5,  "AnalogueGain": 1, "LensPosition": 7.5},
    "730":  {"ExposureTime": 200000, "AnalogueGain": 7.5, "LensPosition": 7.5},
    "850":  {"ExposureTime": 10, "AnalogueGain": 0.5, "LensPosition": 7.5},
}

def apply_capture_settings(key: str) -> None:
    s = CAPTURE_SETTINGS[key]

    controls = {
        "AeEnable": False,
        "ExposureTime": int(s["ExposureTime"]),
        "AnalogueGain": float(s["AnalogueGain"]),
        "LensPosition": float(s["LensPosition"]),
    }

    if key == "still":
        controls["AwbEnable"] = True
        controls["Saturation"] = 1.0
    else:
        controls["AwbEnable"] = False
        controls["Saturation"] = 0.0

        cam.set_controls(controls)


serial = i2c(port=1, address=0x3C)
device = ssd1306(serial, width=128, height=64)
device.contrast(255)

WIDTH = device.width
HEIGHT = device.height
font = ImageFont.load_default()

day = date.today()
last_state = gpio.input(button_pin)

scan_counter = 0
press_counter = 0
current_scan_dir = None

LED_SETTLE = 0.5
LED_OFF_GAP = 0.15
CTRL_SETTLE = 0.15  # small delay after changing exposure controls


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _new_scan_dir(scan_num: int) -> str:
    timestamp = datetime.now().strftime("%H%M%S")
    folder = f"{day}-scan{scan_num:03d}-{timestamp}"
    _ensure_dir(folder)
    return folder


def take_still(scan_dir: str, scan_num: int) -> None:
    apply_capture_settings("still")
    time.sleep(CTRL_SETTLE)
    cam.capture_file(os.path.join(scan_dir, f"{day}-scan{scan_num:03d}-still.jpg"))


def sequence(scan_dir: str, scan_num: int) -> None:
    # 450 nm
    gpio.output(pins["n450"], gpio.HIGH)
    time.sleep(LED_SETTLE)
    apply_capture_settings("450")
    time.sleep(CTRL_SETTLE)
    frame = cam.capture_array("main")
    h,w = frame.shape[0]*2//3, frame.shape[1]
    Y = frame[:h, :]
    img = Image.fromarray(Y, mode="L")
    img.save(os.path.join(scan_dir, f"{day}-scan{scan_num:03d}-450nm.jpg"))
    gpio.output(pins["n450"], gpio.LOW)
    time.sleep(LED_OFF_GAP)

    # 730 nm
    gpio.output(pins["g730"], gpio.HIGH)
    time.sleep(LED_SETTLE)
    apply_capture_settings("730")
    time.sleep(CTRL_SETTLE)
    frame = cam.capture_array("main")
    h,w = frame.shape[0]*2//3, frame.shape[1]
    Y = frame[:h, :]
    img = Image.fromarray(Y, mode="L")
    img.save(os.path.join(scan_dir, f"{day}-scan{scan_num:03d}-730nm.jpg"))
    gpio.output(pins["g730"], gpio.LOW)
    time.sleep(LED_OFF_GAP)

    # 850 nm
    gpio.output(pins["k850"], gpio.HIGH)
    time.sleep(LED_SETTLE)
    apply_capture_settings("850")
    time.sleep(CTRL_SETTLE)
    frame = cam.capture_array("main")
    h,w = frame.shape[0]*2//3, frame.shape[1]
    Y = frame[:h, :]
    img = Image.fromarray(Y, mode="L")
    img.save(os.path.join(scan_dir, f"{day}-scan{scan_num:03d}-850nm.jpg"))
    gpio.output(pins["k850"], gpio.LOW)
    time.sleep(LED_OFF_GAP)


def main():
    global last_state, scan_counter, press_counter, current_scan_dir

    try:
        while True:
            state = gpio.input(button_pin)

            if state == gpio.LOW and last_state == gpio.HIGH:
                press_counter += 1
                next_scan_num = scan_counter + 1

                if press_counter % 2 == 1:
                    current_scan_dir = _new_scan_dir(next_scan_num)
                    take_still(current_scan_dir, next_scan_num)
                else:
                    if not current_scan_dir:
                        current_scan_dir = _new_scan_dir(next_scan_num)
                        take_still(current_scan_dir, next_scan_num)

                    sequence(current_scan_dir, next_scan_num)
                    scan_counter += 1
                    current_scan_dir = None

            last_state = state

            image = Image.new("1", (WIDTH, HEIGHT))
            draw = ImageDraw.Draw(image)

            draw.text((0, 0), "MSI Scanner", font=font, fill=255)
            draw.text((0, 16), f"Scan #: {scan_counter}", font=font, fill=255)

            phase = "STILL" if (press_counter % 2 == 0) else "SCAN"
            draw.text((0, 32), f"Next: {phase}", font=font, fill=255)

            btn_txt = "PRESSED" if state == gpio.LOW else "released"
            draw.text((0, 48), f"Button: {btn_txt}", font=font, fill=255)

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
