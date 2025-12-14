import os
import time
import json
import numpy as np
import RPi.GPIO as gpio

from datetime import date, datetime
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

# =========================
# CONFIG
# =========================

# Save 16-bit + NPY for training
SAVE_NPY = True
SAVE_16BIT_PNG = True

# Capture multiple frames and median-stack for noise reduction
FRAMES_PER_BAND = 5

# Use a fixed center crop to remove housing edges & enforce consistent ROI
USE_CENTER_CROP = True
CROP_SIZE = 900  # pixels (square). Tune to your setup.

# Raw bit depth coming from SRGGB10
RAW_BIT_DEPTH = 10
RAW_MAX = (1 << RAW_BIT_DEPTH) - 1  # 1023

CTRL_SETTLE = 0.12
LED_SETTLE  = 0.35
LED_OFF_GAP = 0.10

# =========================
# GPIO
# =========================
gpio.setmode(gpio.BCM)

pins = {
    "w730": 22,  # 730nm
    "w450": 17,  # 450nm
    "w660": 23,  # 660nm
}

button_pin = 26

for p in pins.values():
    gpio.setup(p, gpio.OUT)
    gpio.output(p, gpio.LOW)

gpio.setup(button_pin, gpio.IN, pull_up_down=gpio.PUD_UP)

# =========================
# CAMERA
# =========================
cam = Picamera2()
cam.configure(cam.create_still_configuration(
    main={"format": "YUV420"},
    raw={"format": "SRGGB10"}
))
cam.start()
time.sleep(0.8)

# Lock focus + kill auto/processing
cam.set_controls({
    "AfMode": 0,
    "LensPosition": 7.5,

    "AeEnable": False,
    "AwbEnable": False,

    "Brightness": 0.0,
    "Contrast": 1.0,
    "Saturation": 0.0,
    "Sharpness": 0.0,
    "NoiseReductionMode": 0,
})

# Per-band exposure/gain (you MUST tune these—450 typically needs more)
CAPTURE_SETTINGS = {
    "still": {"ExposureTime": 3500,  "AnalogueGain": 1.0, "LensPosition": 7.5},

    "w730":  {"ExposureTime": 25000, "AnalogueGain": 2.0, "LensPosition": 7.5},
    "w450":  {"ExposureTime": 45000, "AnalogueGain": 4.0, "LensPosition": 7.5},
    "w660":  {"ExposureTime": 25000, "AnalogueGain": 2.0, "LensPosition": 7.5},
}

def apply_capture_settings(key: str) -> dict:
    s = CAPTURE_SETTINGS[key]
    cam.set_controls({
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": int(s["ExposureTime"]),
        "AnalogueGain": float(s["AnalogueGain"]),
        "LensPosition": float(s["LensPosition"]),
    })
    return {"ExposureTime": int(s["ExposureTime"]), "AnalogueGain": float(s["AnalogueGain"]), "LensPosition": float(s["LensPosition"])}

# =========================
# OLED
# =========================
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial, width=128, height=64)
device.contrast(255)

WIDTH = device.width
HEIGHT = device.height
font = ImageFont.load_default()

# =========================
# IO HELPERS
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _new_scan_dir(scan_num: int) -> str:
    timestamp = datetime.now().strftime("%H%M%S")
    folder = f"{date.today()}-scan{scan_num:03d}-{timestamp}"
    _ensure_dir(folder)
    return folder

def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    if not USE_CENTER_CROP:
        return img
    h, w = img.shape[:2]
    size = min(size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0+size, x0:x0+size]

def _split_rggb_planes(raw: np.ndarray):
    """SRGGB: R at [0,0], B at [1,1]"""
    raw = raw.astype(np.float32)
    R  = raw[0::2, 0::2]
    G1 = raw[0::2, 1::2]
    G2 = raw[1::2, 0::2]
    B  = raw[1::2, 1::2]
    G = (G1 + G2) * 0.5
    return R, G, B

def _raw_to_intensity(raw: np.ndarray) -> np.ndarray:
    """
    Make a single "intensity" image from Bayer planes.
    This avoids the 'pick one plane' trap and is more stable across wavelengths.
    """
    R, G, B = _split_rggb_planes(raw)
    # Simple stable intensity (not normalized): weighted-ish average
    I = (R + 2.0*G + B) * 0.25
    I = _center_crop(I, CROP_SIZE)
    return I

def _to_u16(img: np.ndarray) -> np.ndarray:
    """
    Map expected 10-bit domain -> 16-bit WITHOUT per-image min/max scaling.
    Keeps true intensity relationships for ML.
    """
    x = np.clip(img, 0, RAW_MAX)
    return (x * (65535.0 / RAW_MAX)).astype(np.uint16)

def _save_outputs(out_base: str, img_f32: np.ndarray, meta: dict) -> None:
    # Save 16-bit PNG
    if SAVE_16BIT_PNG:
        u16 = _to_u16(img_f32)
        Image.fromarray(u16, mode="I;16").save(out_base + ".png")

    # Save NPY for training
    if SAVE_NPY:
        np.save(out_base + ".npy", img_f32.astype(np.float32))

    # Save metadata
    with open(out_base + ".json", "w") as f:
        json.dump(meta, f, indent=2)

# =========================
# CAPTURE CORE
# =========================
def _capture_raw_once() -> np.ndarray:
    raw = cam.capture_array("raw")
    return raw

def capture_band_stack(key: str, out_base: str, led_pin: int) -> None:
    """
    Captures:
      - a DARK frame stack (LED off) -> median -> D
      - an ON frame stack (LED on)   -> median -> I
      - returns (I - D) as the saved data (dark-subtracted)
    """
    ctrl = apply_capture_settings(key)
    time.sleep(CTRL_SETTLE)

    # ---- Dark stack (LED OFF) ----
    gpio.output(led_pin, gpio.LOW)
    time.sleep(LED_OFF_GAP)

    dark_frames = []
    for _ in range(FRAMES_PER_BAND):
        raw = _capture_raw_once()
        dark_frames.append(_raw_to_intensity(raw))
        time.sleep(0.01)
    D = np.median(np.stack(dark_frames, axis=0), axis=0)

    # ---- On stack (LED ON) ----
    gpio.output(led_pin, gpio.HIGH)
    time.sleep(LED_SETTLE)

    on_frames = []
    for _ in range(FRAMES_PER_BAND):
        raw = _capture_raw_once()
        on_frames.append(_raw_to_intensity(raw))
        time.sleep(0.01)
    I = np.median(np.stack(on_frames, axis=0), axis=0)

    gpio.output(led_pin, gpio.LOW)
    time.sleep(LED_OFF_GAP)

    # Dark subtraction (cheap win)
    X = I - D
    X = np.clip(X, 0, RAW_MAX)  # keep in sane domain

    meta = {
        "band": key,
        "frames_per_band": FRAMES_PER_BAND,
        "dark_subtraction": True,
        "crop": {"enabled": USE_CENTER_CROP, "size": CROP_SIZE},
        "camera_controls": ctrl,
        "raw_bit_depth": RAW_BIT_DEPTH,
        "timestamp": datetime.now().isoformat(),
    }
    _save_outputs(out_base, X, meta)

def take_still(scan_dir: str, scan_num: int) -> None:
    apply_capture_settings("still")
    time.sleep(CTRL_SETTLE)
    cam.capture_file(os.path.join(scan_dir, f"{date.today()}-scan{scan_num:03d}-still.jpg"))

def sequence(scan_dir: str, scan_num: int) -> None:
    # ORDER: 730, 450, 660
    base = os.path.join(scan_dir, f"{date.today()}-scan{scan_num:03d}")

    capture_band_stack("w730", base + "-730nm", pins["w730"])
    capture_band_stack("w450", base + "-450nm", pins["w450"])
    capture_band_stack("w660", base + "-660nm", pins["w660"])

# =========================
# MAIN LOOP
# =========================
last_state = gpio.input(button_pin)

scan_counter = 0
press_counter = 0
current_scan_dir = None

def oled_status(state: int, scan_counter: int, press_counter: int):
    image = Image.new("1", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), "MSI Scanner", font=font, fill=255)
    draw.text((0, 16), f"Scan #: {scan_counter}", font=font, fill=255)

    next_phase = "STILL" if (press_counter % 2 == 0) else "SCAN"
    draw.text((0, 32), f"Next: {next_phase}", font=font, fill=255)

    btn_txt = "PRESSED" if state == gpio.LOW else "released"
    draw.text((0, 48), f"Button: {btn_txt}", font=font, fill=255)

    device.display(image)

def main():
    global last_state, scan_counter, press_counter, current_scan_dir

    try:
        while True:
            state = gpio.input(button_pin)

            # press edge: HIGH -> LOW
            if state == gpio.LOW and last_state == gpio.HIGH:
                press_counter += 1
                next_scan_num = scan_counter + 1

                if press_counter % 2 == 1:
                    # First press: create folder + still
                    current_scan_dir = _new_scan_dir(next_scan_num)
                    take_still(current_scan_dir, next_scan_num)
                else:
                    # Second press: capture MSI sequence
                    if not current_scan_dir:
                        current_scan_dir = _new_scan_dir(next_scan_num)
                        take_still(current_scan_dir, next_scan_num)

                    sequence(current_scan_dir, next_scan_num)
                    scan_counter += 1
                    current_scan_dir = None

            last_state = state
            oled_status(state, scan_counter, press_counter)
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        device.display(Image.new("1", (WIDTH, HEIGHT)))
        cam.stop()
        gpio.cleanup()

if __name__ == "__main__":
    main()
