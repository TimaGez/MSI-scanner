import os
import time
import json
import signal
import atexit
import threading
import numpy as np
import RPi.GPIO as gpio

from datetime import date, datetime
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

# ============================================================
# GUARANTEES / LIMITS
# ============================================================
MAX_SCAN_SECONDS = 25.0        # total sequence cap
FRAMES_PER_BAND = 2            # keep 2; we’ll fix stalls properly
CAPTURE_TIMEOUT_S = 1.2        # per raw frame capture timeout
CAPTURE_RETRIES = 2            # retry a blocked capture a couple times

USE_CENTER_CROP = True
CROP_SIZE = 700

SAVE_NPY_FLOAT16 = True
SAVE_PREVIEW_PNG = True
SAVE_16BIT_PNG = False

CTRL_SETTLE = 0.04
LED_SETTLE  = 0.02             # small; we pulse per frame now
LED_OFF_GAP = 0.01
INTER_FRAME_GAP = 0.002

RAW_BIT_DEPTH = 10
RAW_MAX = (1 << RAW_BIT_DEPTH) - 1  # 1023
PEDESTAL_RAW = 30.0

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
# HELPERS
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
    R, G, B = _split_rggb_planes(raw)
    I = (R + 2.0*G + B) * 0.25
    return _center_crop(I, CROP_SIZE)

def _to_u16(img: np.ndarray) -> np.ndarray:
    x = np.clip(img, 0, RAW_MAX)
    return (x * (65535.0 / RAW_MAX)).astype(np.uint16)

def _save_preview_png(out_base: str, img_f32: np.ndarray):
    lo, hi = np.percentile(img_f32, (2, 98))
    if hi <= lo:
        vis = np.zeros_like(img_f32, dtype=np.uint8)
    else:
        vis = np.clip((img_f32 - lo) / (hi - lo), 0, 1)
        vis = (vis * 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(out_base + "-preview.png")

def _save_outputs(out_base: str, img_f32: np.ndarray, meta: dict) -> None:
    if SAVE_NPY_FLOAT16:
        np.save(out_base + ".npy", img_f32.astype(np.float16))
    if SAVE_16BIT_PNG:
        u16 = _to_u16(img_f32)
        Image.fromarray(u16, mode="I;16").save(out_base + ".png")
    if SAVE_PREVIEW_PNG:
        _save_preview_png(out_base, img_f32)
    with open(out_base + ".json", "w") as f:
        json.dump(meta, f, indent=2)

# ============================================================
# TIMEOUTS
# ============================================================
class ScanTimeout(Exception):
    pass

class FrameTimeout(Exception):
    pass

def _check_deadline(deadline_t: float):
    if time.monotonic() > deadline_t:
        raise ScanTimeout("Scan exceeded MAX_SCAN_SECONDS")

# ============================================================
# SAFE FRAME CAPTURE (prevents hanging forever)
# ============================================================
def _capture_raw_with_timeout(timeout_s: float) -> np.ndarray:
    result = {}
    exc = {}

    def worker():
        try:
            result["raw"] = cam.capture_array("raw")
        except Exception as e:
            exc["e"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        raise FrameTimeout(f"cam.capture_array('raw') exceeded {timeout_s}s")

    if "e" in exc:
        raise exc["e"]

    return result["raw"]

def _capture_intensity_frame(deadline_t: float) -> np.ndarray:
    _check_deadline(deadline_t)
    raw = _capture_raw_with_timeout(CAPTURE_TIMEOUT_S)
    return _raw_to_intensity(raw).astype(np.float32)

# ============================================================
# CAPTURE CORE (LED PULSED PER FRAME)
# ============================================================
def _avg_stack_pulsed(n: int, led_pin: int, led_on: bool, deadline_t: float) -> (np.ndarray, list):
    """
    Captures n frames. If led_on=True, LED is pulsed for each frame capture
    to prevent it staying on during any unexpected stall.
    Returns average + list of per-frame capture durations.
    """
    acc = None
    durations = []

    for _ in range(n):
        _check_deadline(deadline_t)

        if led_on:
            gpio.output(led_pin, gpio.HIGH)
            time.sleep(LED_SETTLE)
        else:
            gpio.output(led_pin, gpio.LOW)
            time.sleep(LED_OFF_GAP)

        t0 = time.monotonic()
        frame = None

        # retry if capture blocks
        for attempt in range(CAPTURE_RETRIES + 1):
            try:
                frame = _capture_intensity_frame(deadline_t)
                break
            except FrameTimeout:
                # turn LED off immediately and retry
                gpio.output(led_pin, gpio.LOW)
                time.sleep(LED_OFF_GAP)
                if attempt == CAPTURE_RETRIES:
                    raise
                time.sleep(0.02)

        # LED off immediately after frame
        gpio.output(led_pin, gpio.LOW)
        time.sleep(LED_OFF_GAP)

        dt = time.monotonic() - t0
        durations.append(dt)

        if acc is None:
            acc = frame
        else:
            acc += frame

        time.sleep(INTER_FRAME_GAP)

    return acc / float(n), durations

def capture_band(key: str, out_base: str, led_pin: int, deadline_t: float) -> None:
    ctrl = apply_capture_settings(key)
    time.sleep(CTRL_SETTLE)
    _check_deadline(deadline_t)

    # DARK average (LED OFF, pulsed off anyway)
    D, d_times = _avg_stack_pulsed(FRAMES_PER_BAND, led_pin, led_on=False, deadline_t=deadline_t)

    # ON average (LED pulsed per frame)
    I, i_times = _avg_stack_pulsed(FRAMES_PER_BAND, led_pin, led_on=True, deadline_t=deadline_t)

    X = (I - D) + PEDESTAL_RAW
    X = np.clip(X, 0, RAW_MAX)

    meta = {
        "band": key,
        "frames_per_band": FRAMES_PER_BAND,
        "capture_timeout_s": CAPTURE_TIMEOUT_S,
        "capture_retries": CAPTURE_RETRIES,
        "dark_subtraction": True,
        "pedestal_raw": PEDESTAL_RAW,
        "crop": {"enabled": USE_CENTER_CROP, "size": CROP_SIZE},
        "camera_controls": ctrl,
        "raw_bit_depth": RAW_BIT_DEPTH,
        "timings": {
            "dark_frame_seconds": d_times,
            "on_frame_seconds": i_times,
            "dark_total_s": float(sum(d_times)),
            "on_total_s": float(sum(i_times)),
        },
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
        capture_band("w730", base + "-730nm", pins["w730"], deadline_t)
        capture_band("w450", base + "-450nm", pins["w450"], deadline_t)
        capture_band("w660", base + "-660nm", pins["w660"], deadline_t)
    except ScanTimeout:
        all_leds_off()
        with open(base + "-TIMEOUT.txt", "w") as f:
            f.write(f"Timed out after {MAX_SCAN_SECONDS} seconds\n")
        raise
    except FrameTimeout as e:
        all_leds_off()
        with open(base + "-FRAME_TIMEOUT.txt", "w") as f:
            f.write(str(e) + "\n")
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

            if state == gpio.LOW and last_state == gpio.HIGH:
                press_counter += 1
                next_scan_num = scan_counter + 1

                if press_counter % 2 == 1:
                    current_scan_dir = _new_scan_dir(next_scan_num)
                    oled_status(state, scan_counter, press_counter, "STILL")
                    take_still(current_scan_dir, next_scan_num)
                    oled_status(state, scan_counter, press_counter, "READY")
                else:
                    if not current_scan_dir:
                        current_scan_dir = _new_scan_dir(next_scan_num)
                        take_still(current_scan_dir, next_scan_num)

                    oled_status(state, scan_counter, press_counter, "SCAN...")
                    try:
                        sequence(current_scan_dir, next_scan_num)
                        scan_counter += 1
                        oled_status(state, scan_counter, press_counter, "DONE")
                    except Exception:
                        oled_status(state, scan_counter, press_counter, "ERR")
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
