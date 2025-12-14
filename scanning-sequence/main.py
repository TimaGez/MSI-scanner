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
# GOAL: LED/capture <= 20s, processing can take longer
# ============================================================
MAX_SCAN_SECONDS = 20.0     # ONLY applies to capture (LED time)
FRAMES_PER_BAND = 1         # with your current slow RAW capture, 1 is necessary

USE_CENTER_CROP = True
CROP_SIZE = 700

SAVE_NPY_FLOAT16 = True
SAVE_PREVIEW_PNG = True
SAVE_16BIT_PNG = False

PEDESTAL_RAW = 30.0

CTRL_SETTLE = 0.04
LED_SETTLE  = 0.06
LED_OFF_GAP = 0.02

RAW_BIT_DEPTH = 10
RAW_MAX = (1 << RAW_BIT_DEPTH) - 1  # 1023

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
# OLED (FIXED: retry + boot splash + idle refresh)
# ============================================================
font = ImageFont.load_default()

def init_oled_with_retry(max_tries: int = 12, delay_s: float = 0.35):
    """
    This is the missing piece vs your old code.
    Boot can race I2C; retry makes it reliable.
    """
    last_err = None
    for i in range(1, max_tries + 1):
        try:
            serial = i2c(port=1, address=0x3C)  # if your i2cdetect shows 0x3D, change this
            dev = ssd1306(serial, width=128, height=64)
            dev.contrast(255)

            # boot splash
            img = Image.new("1", (dev.width, dev.height))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0),  "MSI Scanner", font=font, fill=255)
            draw.text((0, 16), "Booting...", font=font, fill=255)
            draw.text((0, 32), f"OLED try {i}", font=font, fill=255)
            dev.display(img)

            return dev

        except Exception as e:
            last_err = e
            time.sleep(delay_s)

    print(f"[OLED] failed init after {max_tries} tries: {last_err}")
    return None

device = init_oled_with_retry()
WIDTH = device.width if device else 128
HEIGHT = device.height if device else 64

def oled_msg(line1: str, line2: str = "", line3: str = "", line4: str = ""):
    if device is None:
        return
    img = Image.new("1", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0),  line1[:21], font=font, fill=255)
    draw.text((0, 16), line2[:21], font=font, fill=255)
    draw.text((0, 32), line3[:21], font=font, fill=255)
    draw.text((0, 48), line4[:21], font=font, fill=255)
    device.display(img)

# Immediately show READY on startup (this is also something the last code lacked)
oled_msg("MSI Scanner", "READY", "Press button", "for still/scan")

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
        if device:
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
    R, G, B = _split_rggb_planes(raw)
    I = (R + 2.0*G + B) * 0.25
    return _center_crop(I, CROP_SIZE).astype(np.float32)

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
# CAPTURE (FAST) + PROCESS (AFTER)
# ============================================================
class ScanTimeout(Exception):
    pass

def _deadline_ok(deadline_t: float):
    if time.monotonic() > deadline_t:
        raise ScanTimeout("Scan exceeded MAX_SCAN_SECONDS")

def _capture_n_intensity(n: int) -> (np.ndarray, list):
    acc = None
    times = []
    for _ in range(n):
        t0 = time.monotonic()
        raw = cam.capture_array("raw")
        frame = _raw_to_intensity(raw)
        times.append(time.monotonic() - t0)
        if acc is None:
            acc = frame
        else:
            acc += frame
    return acc / float(n), times

def scan_band_capture_only(key: str, led_pin: int, deadline_t: float):
    ctrl = apply_capture_settings(key)
    time.sleep(CTRL_SETTLE)
    _deadline_ok(deadline_t)

    gpio.output(led_pin, gpio.LOW)
    time.sleep(LED_OFF_GAP)
    _deadline_ok(deadline_t)
    D, d_times = _capture_n_intensity(FRAMES_PER_BAND)

    gpio.output(led_pin, gpio.HIGH)
    time.sleep(LED_SETTLE)
    _deadline_ok(deadline_t)
    I, i_times = _capture_n_intensity(FRAMES_PER_BAND)

    gpio.output(led_pin, gpio.LOW)
    time.sleep(LED_OFF_GAP)

    timing = {
        "dark_frame_seconds": d_times,
        "on_frame_seconds": i_times,
        "dark_total_s": float(sum(d_times)),
        "on_total_s": float(sum(i_times)),
    }
    return D, I, ctrl, timing

def take_still(scan_dir: str, scan_num: int) -> None:
    apply_capture_settings("still")
    time.sleep(CTRL_SETTLE)
    cam.capture_file(os.path.join(scan_dir, f"{date.today()}-scan{scan_num:03d}-still.jpg"))

def sequence(scan_dir: str, scan_num: int) -> None:
    base = os.path.join(scan_dir, f"{date.today()}-scan{scan_num:03d}")
    deadline_t = time.monotonic() + MAX_SCAN_SECONDS

    oled_msg("MSI Scanner", f"Scan {scan_num}", "CAPTURING...", "")
    t_scan0 = time.monotonic()

    captured = {}
    try:
        oled_msg("MSI Scanner", f"Scan {scan_num}", "CAPTURE 730", "")
        captured["w730"] = scan_band_capture_only("w730", pins["w730"], deadline_t)

        oled_msg("MSI Scanner", f"Scan {scan_num}", "CAPTURE 450", "")
        captured["w450"] = scan_band_capture_only("w450", pins["w450"], deadline_t)

        oled_msg("MSI Scanner", f"Scan {scan_num}", "CAPTURE 660", "")
        captured["w660"] = scan_band_capture_only("w660", pins["w660"], deadline_t)

    except ScanTimeout:
        all_leds_off()
        with open(base + "-TIMEOUT.txt", "w") as f:
            f.write(f"Timed out after {MAX_SCAN_SECONDS} seconds\n")
        raise
    finally:
        all_leds_off()

    t_scan = time.monotonic() - t_scan0

    # processing unbounded
    oled_msg("MSI Scanner", f"Scan {scan_num}", "PROCESSING...", "")
    t_proc0 = time.monotonic()

    for key, (D, I, ctrl, timing) in captured.items():
        nm = {"w730": "730nm", "w450": "450nm", "w660": "660nm"}[key]
        out_base = f"{base}-{nm}"

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
            "timings": timing,
            "capture_scan_total_s": float(t_scan),
            "timestamp": datetime.now().isoformat(),
        }
        _save_outputs(out_base, X, meta)
        oled_msg("MSI Scanner", f"Scan {scan_num}", "PROCESSING...", nm)

    t_proc = time.monotonic() - t_proc0
    oled_msg("MSI Scanner", f"Scan {scan_num}", "DONE",
             f"cap:{t_scan:.1f}s", f"proc:{t_proc:.1f}s")

# ============================================================
# MAIN LOOP (FIXED: idle refresh so OLED stays alive)
# ============================================================
last_state = gpio.input(button_pin)
scan_counter = 0
press_counter = 0
current_scan_dir = None
last_idle_draw = 0.0

def main():
    global last_state, scan_counter, press_counter, current_scan_dir, last_idle_draw

    try:
        while True:
            state = gpio.input(button_pin)

            # idle refresh every ~0.5s so OLED doesn't stay blank if first draw missed
            now = time.monotonic()
            if now - last_idle_draw > 0.5:
                next_phase = "STILL" if (press_counter % 2 == 0) else "SCAN"
                oled_msg("MSI Scanner",
                         f"Scan #: {scan_counter}",
                         f"Next: {next_phase}",
                         "Press button")
                last_idle_draw = now

            if state == gpio.LOW and last_state == gpio.HIGH:
                press_counter += 1
                next_scan_num = scan_counter + 1

                if press_counter % 2 == 1:
                    current_scan_dir = _new_scan_dir(next_scan_num)
                    oled_msg("MSI Scanner", f"Scan {next_scan_num}", "STILL...", "")
                    take_still(current_scan_dir, next_scan_num)
                    oled_msg("MSI Scanner", f"Scan {next_scan_num}", "READY", "Press again", "to scan")
                else:
                    if not current_scan_dir:
                        current_scan_dir = _new_scan_dir(next_scan_num)
                        take_still(current_scan_dir, next_scan_num)

                    try:
                        sequence(current_scan_dir, next_scan_num)
                        scan_counter += 1
                    except Exception:
                        oled_msg("MSI Scanner", f"Scan {next_scan_num}", "ERROR", "", "")
                    finally:
                        current_scan_dir = None
                        all_leds_off()

            last_state = state
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        all_leds_off()
        try:
            if device:
                device.display(Image.new("1", (WIDTH, HEIGHT)))
        except Exception:
            pass
        try:
            cam.stop()
        except Exception:
            pass
        gpio.cleanup()

if __name__ == "__main__":
    main()
