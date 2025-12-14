import os
import json
import time
import numpy as np
import RPi.GPIO as gpio
from picamera2 import Picamera2
from datetime import date, datetime
from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

# ---------------- GPIO ----------------
gpio.setmode(gpio.BCM)

pins = {
    "n450": 17,
    "g730": 22,
    "k850": 23,
}

button_pin = 26

for k in pins.values():
    gpio.setup(k, gpio.OUT)
    gpio.output(k, gpio.LOW)  # ensure OFF

gpio.setup(button_pin, gpio.IN, pull_up_down=gpio.PUD_UP)

# ---------------- CAMERA ----------------
cam = Picamera2()

# Force consistent pipeline format for scan captures (YUV420 so we can extract Y cleanly)
cam.configure(cam.create_still_configuration(main={"format": "YUV420"}))
cam.start()

# Let camera settle
time.sleep(0.7)

# Disable AF + set an initial lens position (you can tune per capture later)
cam.set_controls({
    "AfMode": 0,            # manual focus
    "LensPosition": 4.5,
})

# Global ISP controls: keep minimal and stable
cam.set_controls({
    "AeEnable": False,
    "AwbEnable": False,
    "ColourGains": (1.0, 1.0),
    "Saturation": 0.0,          # doesn’t matter for Y plane, but keeps ISP calm
    "Sharpness": 0.0,
    "NoiseReductionMode": 0,    # OFF (important for training consistency)
})

# NOTE: these wavelength labels don't need to be accurate — we're treating them as "bands"
CAPTURE_SETTINGS = {
    # Still is allowed to be a normal-looking color-ish reference image
    "still": {"ExposureTime": 350, "AnalogueGain": 1.0, "LensPosition": 4.5, "AwbEnable": True,  "Saturation": 1.0},

    # Scan bands should be as raw/stable as possible (no AWB, no saturation)
    "450":   {"ExposureTime": 5,       "AnalogueGain": 1.0, "LensPosition": 7.5, "AwbEnable": False, "Saturation": 0.0},
    "730":   {"ExposureTime": 200000,  "AnalogueGain": 7.5, "LensPosition": 7.5, "AwbEnable": False, "Saturation": 0.0},
    "850":   {"ExposureTime": 10,      "AnalogueGain": 0.5, "LensPosition": 7.5, "AwbEnable": False, "Saturation": 0.0},
}

# How many frames to flush after changing controls (ensures settings are applied)
CONTROL_FLUSH_FRAMES = 2

LED_SETTLE = 0.50
LED_OFF_GAP = 0.15
DEBOUNCE_SEC = 0.20

# ---------------- OLED ----------------
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial, width=128, height=64)
device.contrast(255)

WIDTH = device.width
HEIGHT = device.height
font = ImageFont.load_default()

# ---------------- SCAN STATE ----------------
day = date.today()
last_state = gpio.input(button_pin)

scan_counter = 0          # increments only after a full scan sequence (still + bands)
press_counter = 0         # increments on every valid press
current_scan_dir = None
last_press_time = 0.0


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _new_scan_dir(scan_num: int) -> str:
    timestamp = datetime.now().strftime("%H%M%S")
    folder = f"{day}-scan{scan_num:03d}-{timestamp}"
    _ensure_dir(folder)
    return folder


def apply_capture_settings(key: str) -> None:
    s = CAPTURE_SETTINGS[key]
    controls = {
        "AeEnable": False,
        "ExposureTime": int(s["ExposureTime"]),
        "AnalogueGain": float(s["AnalogueGain"]),
        "AfMode": 0,
        "LensPosition": float(s["LensPosition"]),
        "AwbEnable": bool(s["AwbEnable"]),
        "Saturation": float(s["Saturation"]),
        "NoiseReductionMode": 0,
        "Sharpness": 0.0,
    }
    cam.set_controls(controls)

    # Flush frames so controls actually take effect on the captured frame
    for _ in range(CONTROL_FLUSH_FRAMES):
        cam.capture_array("main")


def capture_band_y_png(path_png: str) -> dict:
    """
    Captures the luminance (Y plane) as an 8-bit grayscale PNG.
    Returns metadata dict for reproducibility / training.
    """
    frame = cam.capture_array("main")  # YUV420 buffer in a 2D array
    h = frame.shape[0] * 2 // 3        # Y plane height for YUV420
    Y = frame[:h, :]

    # Save lossless. (JPG is bad for training, especially on low-contrast details)
    img = Image.fromarray(Y, mode="L")
    img.save(path_png)

    meta = cam.capture_metadata()
    return {
        "file": os.path.basename(path_png),
        "ExposureTime": meta.get("ExposureTime"),
        "AnalogueGain": meta.get("AnalogueGain"),
        "LensPosition": meta.get("LensPosition"),
        "FrameDuration": meta.get("FrameDuration"),
        "SensorTimestamp": meta.get("SensorTimestamp"),
        "shape": [int(Y.shape[0]), int(Y.shape[1])],
        "dtype": str(Y.dtype),
    }


def take_still(scan_dir: str, scan_num: int, meta_log: dict) -> None:
    apply_capture_settings("still")

    still_path = os.path.join(scan_dir, f"{day}-scan{scan_num:03d}-still.jpg")
    cam.capture_file(still_path)

    m = cam.capture_metadata()
    meta_log["still"] = {
        "file": os.path.basename(still_path),
        "ExposureTime": m.get("ExposureTime"),
        "AnalogueGain": m.get("AnalogueGain"),
        "LensPosition": m.get("LensPosition"),
        "AwbApplied": True,
        "SensorTimestamp": m.get("SensorTimestamp"),
    }


def sequence(scan_dir: str, scan_num: int, meta_log: dict) -> None:
    meta_log["bands"] = {}

    # --- BAND: 450 ---
    gpio.output(pins["n450"], gpio.HIGH)
    time.sleep(LED_SETTLE)
    apply_capture_settings("450")
    p = os.path.join(scan_dir, f"{day}-scan{scan_num:03d}-450.png")
    meta_log["bands"]["450"] = capture_band_y_png(p)
    gpio.output(pins["n450"], gpio.LOW)
    time.sleep(LED_OFF_GAP)

    # --- BAND: 730 ---
    gpio.output(pins["g730"], gpio.HIGH)
    time.sleep(LED_SETTLE)
    apply_capture_settings("730")
    p = os.path.join(scan_dir, f"{day}-scan{scan_num:03d}-730.png")
    meta_log["bands"]["730"] = capture_band_y_png(p)
    gpio.output(pins["g730"], gpio.LOW)
    time.sleep(LED_OFF_GAP)

    # --- BAND: 850 ---
    gpio.output(pins["k850"], gpio.HIGH)
    time.sleep(LED_SETTLE)
    apply_capture_settings("850")
    p = os.path.join(scan_dir, f"{day}-scan{scan_num:03d}-850.png")
    meta_log["bands"]["850"] = capture_band_y_png(p)
    gpio.output(pins["k850"], gpio.LOW)
    time.sleep(LED_OFF_GAP)


def write_meta(scan_dir: str, meta_log: dict) -> None:
    meta_path = os.path.join(scan_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_log, f, indent=2)


def main():
    global last_state, scan_counter, press_counter, current_scan_dir, last_press_time

    try:
        while True:
            state = gpio.input(button_pin)

            # detect press edge
            if state == gpio.LOW and last_state == gpio.HIGH:
                now = time.monotonic()
                if now - last_press_time < DEBOUNCE_SEC:
                    # ignore bounce / accidental double press
                    last_state = state
                    continue
                last_press_time = now

                press_counter += 1
                next_scan_num = scan_counter + 1

                if press_counter % 2 == 1:
                    # odd press: start a new scan folder + still image
                    current_scan_dir = _new_scan_dir(next_scan_num)

                    meta_log = {
                        "scan_num": next_scan_num,
                        "date": str(day),
                        "created": datetime.now().isoformat(),
                        "press_counter": press_counter,
                        "notes": "Odd press = still, even press = bands. Bands saved as PNG (lossless).",
                    }

                    take_still(current_scan_dir, next_scan_num, meta_log)
                    write_meta(current_scan_dir, meta_log)

                else:
                    # even press: run spectral sequence in the existing folder
                    if not current_scan_dir:
                        current_scan_dir = _new_scan_dir(next_scan_num)

                    # load existing meta if present, else create
                    meta_path = os.path.join(current_scan_dir, "meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r") as f:
                            meta_log = json.load(f)
                    else:
                        meta_log = {
                            "scan_num": next_scan_num,
                            "date": str(day),
                            "created": datetime.now().isoformat(),
                            "press_counter": press_counter,
                        }

                    sequence(current_scan_dir, next_scan_num, meta_log)
                    meta_log["completed"] = datetime.now().isoformat()
                    write_meta(current_scan_dir, meta_log)

                    scan_counter += 1
                    current_scan_dir = None

            last_state = state

            # OLED UI
            image = Image.new("1", (WIDTH, HEIGHT))
            draw = ImageDraw.Draw(image)

            draw.text((0, 0), "MSI Scanner", font=font, fill=255)
            draw.text((0, 16), f"Scan #: {scan_counter}", font=font, fill=255)

            next_action = "STILL" if (press_counter % 2 == 0) else "SCAN"
            draw.text((0, 32), f"Next: {next_action}", font=font, fill=255)

            btn_txt = "PRESSED" if state == gpio.LOW else "released"
            draw.text((0, 48), f"Button: {btn_txt}", font=font, fill=255)

            device.display(image)
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        # turn off LEDs
        for k in pins.values():
            gpio.output(k, gpio.LOW)

        # clear screen
        image = Image.new("1", (WIDTH, HEIGHT))
        device.display(image)

        cam.stop()
        gpio.cleanup()


if __name__ == "__main__":
    main()
