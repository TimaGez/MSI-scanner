import os
import time
import json
import signal
import atexit
import numpy as np
import RPi.GPIO as gpio

from datetime import date, datetime
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

# ============================================================
# SPEED / RELIABILITY CONFIG
# ============================================================

# Hard cap. This code is faster, so it shouldn't hit it.
# If you REALLY need max 20, set to 20.0 after you confirm it runs.
MAX_SCAN_SECONDS = 25.0

# Capture count per band:
# 2 gives a nice noise reduction without costing time/RAM.
FRAMES_PER_BAND = 2

# Cropping makes everything faster + removes housing edges
USE_CENTER_CROP = True
CROP_SIZE = 700  # smaller = faster disk + faster processing (tune)

# Output strategy (fast):
# - Save .npy in float16 (small + fast, still great for ML)
# - Save preview PNG (8-bit stretched) for humans
SAVE_NPY_FLOAT16 = True
SAVE_PREVIEW_PNG = True

# Optional "scientific" 16-bit PNG (often big + slow on SD card)
SAVE_16BIT_PNG = False

# Fast settles
CTRL_SETTLE = 0.04
LED_SETTLE  = 0.06
LED_OFF_GAP = 0.02
INTER_FRAME_GAP = 0.001

RAW_BIT_DEPTH = 10
RAW_MAX = (1 << RAW_BIT_DEPTH) - 1  # 1023

# After dark subtraction, add a small pedestal to avoid "all zeros"
# (helps stability + prevents flat previews)
PEDESTAL_RAW = 30.0  # tune 0–80

# ============================================================
# GPIO
# ============================================================
gpio.setmode(gpio.BCM)

pins = {
    "w730": 22,
    "w450": 17,
    "w660": 23,
}
button_pin = 26

for p in pins.values():
    gpio.setup(p, gpio.OUT)
    gpio.output(p, gpio.LOW)

gpio.setup(button_pin, gpio.IN, pull_up_down=gpio.PUD_UP)

def all_leds_off():
    for p in pins.values():
        try:
            gpio.output(p, gpio.LOW)
        except Exception:
            pass

atexit.register(all_leds_off)

# ============================================================
# CAMERA
# ============================================================
cam = Picamera2()
cam.configure(cam.create_still_configuration(
    main={"format": "YUV420"},
    raw={"format": "SRGGB10"}
))
cam.start()
time.sleep(0.6)

# Lock focus + disable auto + disable extra processing
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

# Per-band settings (you will tune these)
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
    return {
        "ExposureTime": int(s["ExposureTime"]),
        "AnalogueGain": float(s["AnalogueGain"]),
        "LensPosition": float(s["LensPosition"]),
    }

# ============================================================
# OLED
# ============================================================
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial, width=128, height=64)
device.contrast(255)

WIDTH = device.width
HEIGHT = device.height
font = ImageFont.load_default()

# ============================================================
# SIGNAL SAFETY
# ============================================================
def _handle_signal(signum, frame):
    all_leds_off()
    try:
        cam.stop()
    except Exception:
        pass
    try:
        device.display(Image.new("1", (WIDTH, HEIGHT)))
    except Exception:
        pass
    try:
        gpio.cleanup()
    except Exception:
        pass
    raise SystemExit(0)

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ============================================================
# FILE / IMAGE HELPERS
# ============================================================
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
    raw = raw.astype(np.float32)
    R  = raw[0::2, 0::2]
    G1 = raw[0::2, 1::2]
    G2 = raw[1::2, 0::2]
    B  = raw[1::2, 1::2]
    G = (G1 + G2) * 0.5
    return R, G, B

def _raw_to_intensity(raw: np.ndarray) -> np.ndarray:
    # Stable intensity (avoid picking one plane)
    R, G, B = _split_rggb_planes(raw)
    I = (R + 2.0*G + B) * 0.25
    return _center_crop(I, CROP_SIZE)

def _to_u16(img: np.ndarray) -> np.ndarray:
    x = np.clip(img, 0, RAW_MAX)
    return (x * (65535.0 / RAW_MAX)).astype(np.uint16)

def _save_preview_png(out_base: str, img_f32: np.ndarray):
    # Percentile stretch for viewing (does NOT affect training array)
    lo, hi = np.percentile(img_f32, (2, 98))
    if hi <= lo:
        vis = np.zeros_like(img_f32, dtype=np.uint8)
    else:
        vis = np.clip((img_f32 - lo) / (hi - lo), 0, 1)
        vis = (vis * 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(out_base + "-preview.png")

def _save_outputs(out_base: str, img_f32: np.ndarray, meta: dict) -> None:
    # Fast ML tensor
    if SAVE_NPY_FLOAT16:
        np.save(out_base + ".npy", img_f32.astype(np.float16))

    # Optional 16-bit png (slow)
    if SAVE_16BIT_PNG:
        u16 = _to_u16(img_f32)
        Image.fromarray(u16, mode="I;16").save(out_base + ".png")

    # Preview for humans
    if SAVE_PREVIEW_PNG:
        _save_preview_png(out_base, img_f32)

    with open(out_base + ".json", "w") as f:
        json.dump(meta, f, indent=2)

# ============================================================
# TIMEOUT / DEADLINE
# ============================================================
class ScanTimeout(Exception):
    pass

def _check_deadline(deadline_t: float):
    if time.monotonic() > deadline_t:
        raise ScanTimeout("Scan exceeded MAX_SCAN_SECONDS")

# ============================================================
# CAPTURE CORE (FAST + LED OFF BEFORE SAVING)
# ============================================================
def _avg_stack(n: int, deadline_t: float) -> np.ndarray:
    acc = None
    for _ in range(n):
        _check_deadline(deadline_t)
        raw = cam.capture_array("raw")
        frame = _raw_to_intensity(raw)  # float32
        if acc is None:
            acc = frame
        else:
            acc += frame
        time.sleep(INTER_FRAME_GAP)
    return acc / float(n)

def capture_band_fast(key: str, out_base: str, led_pin: int, deadline_t: float) -> None:
    ctrl = apply_capture_settings(key)
    time.sleep(CTRL_SETTLE)
    _check_deadline(deadline_t)

    # Make absolutely sure LED is off before starting dark frames
    gpio.output(led_pin, gpio.LOW)
    time.sleep(LED_OFF_GAP)

    # DARK frames (LED OFF)
    D = _avg_stack(FRAMES_PER_BAND, deadline_t)

    # ON frames (LED ON)
    gpio.output(led_pin, gpio.HIGH)
    time.sleep(LED_SETTLE)

    I = _avg_stack(FRAMES_PER_BAND, deadline_t)

    # IMPORTANT FIX:
    # Turn LED OFF IMMEDIATELY after captures, BEFORE processing/saving.
    gpio.output(led_pin, gpio.LOW)
    time.sleep(LED_OFF_GAP)

    # Now do math + saving with LED safely off
    X = (I - D) + PEDESTAL_RAW
    X = np.clip(X, 0, RAW_MAX)

    meta = {
        "band": key,
        "frames_per_band": FRAMES_PER_BAND,
        "dark_subtraction": True,
        "pedestal_raw": PEDESTAL_RAW,
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
    deadline_t = time.monotonic() + MAX_SCAN_SECONDS
    base = os.path.join(scan_dir, f"{date.today()}-scan{scan_num:03d}")

    try:
        capture_band_fast("w730", base + "-730nm", pins["w730"], deadline_t)
        capture_band_fast("w450", base + "-450nm", pins["w450"], deadline_t)
        capture_band_fast("w660", base + "-660nm", pins["w660"], deadline_t)

    except ScanTimeout:
        # Force all LEDs off and log timeout
        all_leds_off()
        with open(base + "-TIMEOUT.txt", "w") as f:
            f.write(f"Timed out after {MAX_SCAN_SECONDS} seconds\n")
        raise

# ============================================================
# MAIN LOOP
# ============================================================
last_state = gpio.input(button_pin)
scan_counter = 0
press_counter = 0
current_scan_dir = None

def oled_status(state: int, scan_counter: int, press_counter: int, msg: str = ""):
    image = Image.new("1", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)

    draw.text((0, 0), "MSI Scanner", font=font, fill=255)
    draw.text((0, 16), f"Scan #: {scan_counter}", font=font, fill=255)
    next_phase = "STILL" if (press_counter % 2 == 0) else "SCAN"
    draw.text((0, 32), f"Next: {next_phase}", font=font, fill=255)

    btn_txt = "PRESSED" if state == gpio.LOW else "released"
    draw.text((0, 48), (btn_txt + " " + msg)[:21], font=font, fill=255)
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
                    # Press 1: make folder + still
                    current_scan_dir = _new_scan_dir(next_scan_num)
                    oled_status(state, scan_counter, press_counter, "STILL")
                    take_still(current_scan_dir, next_scan_num)
                    oled_status(state, scan_counter, press_counter, "READY")

                else:
                    # Press 2: scan
                    if not current_scan_dir:
                        current_scan_dir = _new_scan_dir(next_scan_num)
                        take_still(current_scan_dir, next_scan_num)

                    oled_status(state, scan_counter, press_counter, "SCAN...")
                    try:
                        sequence(current_scan_dir, next_scan_num)
                        scan_counter += 1
                        oled_status(state, scan_counter, press_counter, "DONE")
                    except ScanTimeout:
                        oled_status(state, scan_counter, press_counter, "TIMEOUT")
                    finally:
                        current_scan_dir = None
                        all_leds_off()

            last_state = state
            oled_status(state, scan_counter, press_counter)
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        all_leds_off()
        device.display(Image.new("1", (WIDTH, HEIGHT)))
        try:
            cam.stop()
        except Exception:
            pass
        gpio.cleanup()

if __name__ == "__main__":
    main()
