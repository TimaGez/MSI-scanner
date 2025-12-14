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
# CONFIG
# ============================================================
MAX_CAPTURE_SECONDS = 20.0     # cap ONLY capture/LED time
LONG_PRESS_S = 1.2

FRAMES_PER_BAND = {
    "w730": 1,
    "w660": 1,
    "w450": 2,   # blue needs more averaging
}

USE_CENTER_CROP = True
CROP_SIZE = 700

SAVE_PER_BAND_CORRECTED = True
SAVE_PREVIEW_PNG = True
SAVE_ML_TENSOR = True
SAVE_FLOAT16 = True

CAL_DIR = "calibration"
EPS = 1e-6

PEDESTAL_RAW = 20.0
CLIP_PCT = (1, 99)

CTRL_SETTLE = 0.04
LED_SETTLE  = 0.08
LED_OFF_GAP = 0.02
INTER_FRAME_GAP = 0.002

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
# OLED
# ============================================================
font = ImageFont.load_default()

def init_oled_with_retry(max_tries: int = 12, delay_s: float = 0.35):
    last_err = None
    for i in range(1, max_tries + 1):
        try:
            serial = i2c(port=1, address=0x3C)
            dev = ssd1306(serial, width=128, height=64)
            dev.contrast(255)

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

    print(f"[OLED] init failed: {last_err}")
    return None

device = init_oled_with_retry()
WIDTH = device.width if device else 128
HEIGHT = device.height if device else 64

def oled_msg(a="", b="", c="", d=""):
    if device is None:
        return
    img = Image.new("1", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0),  a[:21], font=font, fill=255)
    draw.text((0, 16), b[:21], font=font, fill=255)
    draw.text((0, 32), c[:21], font=font, fill=255)
    draw.text((0, 48), d[:21], font=font, fill=255)
    device.display(img)

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
# FILE HELPERS
# ============================================================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _new_session_dir(session_num: int) -> str:
    ts = datetime.now().strftime("%H%M%S")
    folder = f"{date.today()}-session{session_num:03d}-{ts}"
    _ensure_dir(folder)
    return folder

def _cal_path(band: str) -> str:
    _ensure_dir(CAL_DIR)
    return os.path.join(CAL_DIR, f"flat_{band}.npy")

def have_calibration() -> bool:
    return all(os.path.exists(_cal_path(b)) for b in ["w730", "w450", "w660"])

# ============================================================
# IMAGE HELPERS
# ============================================================
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

def _clip_percentile(x: np.ndarray, pct=(1, 99)) -> np.ndarray:
    lo, hi = np.percentile(x, pct)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x, lo, hi).astype(np.float32)

def _save_preview_png(path: str, img_f32: np.ndarray):
    lo, hi = np.percentile(img_f32, (2, 98))
    if hi <= lo:
        vis = np.zeros_like(img_f32, dtype=np.uint8)
    else:
        vis = np.clip((img_f32 - lo) / (hi - lo), 0, 1)
        vis = (vis * 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(path)

def _save_npy(path: str, arr: np.ndarray):
    np.save(path, arr.astype(np.float16) if SAVE_FLOAT16 else arr.astype(np.float32))

# ============================================================
# CAPTURE CORE
# ============================================================
class CaptureTimeout(Exception):
    pass

def _deadline_ok(deadline_t: float):
    if time.monotonic() > deadline_t:
        raise CaptureTimeout("Capture exceeded MAX_CAPTURE_SECONDS")

def _capture_avg_intensity(n: int) -> (np.ndarray, list):
    acc = None
    times = []
    for _ in range(n):
        t0 = time.monotonic()
        raw = cam.capture_array("raw")
        frame = _raw_to_intensity(raw)
        times.append(time.monotonic() - t0)
        acc = frame if acc is None else (acc + frame)
        time.sleep(INTER_FRAME_GAP)
    return acc / float(n), times

def capture_band_DI(band: str, led_pin: int, deadline_t: float):
    ctrl = apply_capture_settings(band)
    time.sleep(CTRL_SETTLE)
    _deadline_ok(deadline_t)

    n = FRAMES_PER_BAND[band]

    # DARK
    gpio.output(led_pin, gpio.LOW)
    time.sleep(LED_OFF_GAP)
    _deadline_ok(deadline_t)
    D, d_times = _capture_avg_intensity(n)

    # ON
    gpio.output(led_pin, gpio.HIGH)
    time.sleep(LED_SETTLE)
    _deadline_ok(deadline_t)
    I, i_times = _capture_avg_intensity(n)

    gpio.output(led_pin, gpio.LOW)
    time.sleep(LED_OFF_GAP)

    timing = {
        "frames": n,
        "dark_frame_seconds": d_times,
        "on_frame_seconds": i_times,
        "dark_total_s": float(sum(d_times)),
        "on_total_s": float(sum(i_times)),
    }
    return D, I, ctrl, timing

# ============================================================
# STILL (MUST BE UNCHANGED)
# ============================================================
def take_still_unmodified(session_dir: str, session_num: int) -> str:
    """
    Saves still.jpg EXACTLY as camera outputs (no processing).
    """
    apply_capture_settings("still")
    time.sleep(CTRL_SETTLE)
    path = os.path.join(session_dir, f"{date.today()}-session{session_num:03d}-still.jpg")
    cam.capture_file(path)
    return path

# ============================================================
# WHITE FIELD CAL
# ============================================================
def capture_white_reference(session_dir: str, session_num: int):
    """
    Captures and SAVES flat-field for each band.
    Must be done in same position/pressure as scan.
    """
    oled_msg("CALIBRATION", "White-field", "Capturing...", "")
    deadline_t = time.monotonic() + MAX_CAPTURE_SECONDS

    flats = {}
    for band, nm in [("w730","730"), ("w450","450"), ("w660","660")]:
        oled_msg("CALIBRATION", f"CAP {nm}nm", "", "")
        D, I, ctrl, timing = capture_band_DI(band, pins[band], deadline_t)
        W = (I - D) + PEDESTAL_RAW
        W = np.clip(W, 0, RAW_MAX).astype(np.float32)
        W = _clip_percentile(W, CLIP_PCT)
        flats[band] = (W, ctrl, timing)

    for band in ["w730","w450","w660"]:
        W, _, _ = flats[band]
        _save_npy(_cal_path(band), W)

    # optional save a copy inside session folder too
    for band, nm in [("w730","730nm"), ("w450","450nm"), ("w660","660nm")]:
        W, _, _ = flats[band]
        out = os.path.join(session_dir, f"{date.today()}-session{session_num:03d}-flat-{nm}")
        _save_npy(out + ".npy", W)
        if SAVE_PREVIEW_PNG:
            _save_preview_png(out + "-preview.png", W)

    oled_msg("CALIBRATION", "Saved flats ✅", "", "Ready")

# ============================================================
# SCAN + PROCESS
# ============================================================
def process_and_save(session_dir: str, session_num: int, captured: dict):
    base = os.path.join(session_dir, f"{date.today()}-session{session_num:03d}")
    oled_msg("MSI Scanner", f"Session {session_num}", "PROCESSING...", "")

    flats = None
    if have_calibration():
        flats = {b: np.load(_cal_path(b)).astype(np.float32) for b in ["w730","w450","w660"]}

    corr = {}
    meta = {
        "session_num": session_num,
        "timestamp": datetime.now().isoformat(),
        "frames_per_band": FRAMES_PER_BAND,
        "crop": {"enabled": USE_CENTER_CROP, "size": CROP_SIZE},
        "dark_subtraction": True,
        "flat_field_used": bool(flats is not None),
        "pedestal_raw": PEDESTAL_RAW,
        "clip_percentiles": CLIP_PCT,
        "bands": {},
    }

    for band, nm in [("w730","730nm"), ("w450","450nm"), ("w660","660nm")]:
        D, I, ctrl, timing = captured[band]
        X = (I - D) + PEDESTAL_RAW
        X = np.clip(X, 0, RAW_MAX).astype(np.float32)

        if flats is not None:
            X = X / (flats[band] + EPS)

        X = _clip_percentile(X, CLIP_PCT)
        corr[band] = X

        meta["bands"][band] = {"name": nm, "camera_controls": ctrl, "timings": timing}

        if SAVE_PER_BAND_CORRECTED:
            out_band = f"{base}-{nm}-corr"
            _save_npy(out_band + ".npy", X)
            if SAVE_PREVIEW_PNG:
                _save_preview_png(out_band + "-preview.png", X)

    if SAVE_ML_TENSOR:
        r_660_730 = corr["w660"] / (corr["w730"] + EPS)
        r_450_660 = corr["w450"] / (corr["w660"] + EPS)
        r_450_730 = corr["w450"] / (corr["w730"] + EPS)

        r_660_730 = _clip_percentile(r_660_730, CLIP_PCT)
        r_450_660 = _clip_percentile(r_450_660, CLIP_PCT)
        r_450_730 = _clip_percentile(r_450_730, CLIP_PCT)

        tensor = np.stack([r_660_730, r_450_660, r_450_730], axis=-1).astype(np.float32)
        out_ml = f"{base}-ML-ratios"
        _save_npy(out_ml + ".npy", tensor)

        if SAVE_PREVIEW_PNG:
            _save_preview_png(out_ml + "-ch0-660over730.png", tensor[..., 0])
            _save_preview_png(out_ml + "-ch1-450over660.png", tensor[..., 1])
            _save_preview_png(out_ml + "-ch2-450over730.png", tensor[..., 2])

        meta["ml_tensor"] = {"file": os.path.basename(out_ml + ".npy"),
                             "channels": ["660/730", "450/660", "450/730"]}

    with open(base + "-meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    oled_msg("MSI Scanner", f"Session {session_num}", "DONE ✅", "Saved ML tensor")

def run_scan(session_dir: str, session_num: int):
    deadline_t = time.monotonic() + MAX_CAPTURE_SECONDS
    base = os.path.join(session_dir, f"{date.today()}-session{session_num:03d}")

    captured = {}
    t0 = time.monotonic()
    try:
        oled_msg("MSI Scanner", f"Session {session_num}", "CAPTURE 730", "")
        captured["w730"] = capture_band_DI("w730", pins["w730"], deadline_t)

        oled_msg("MSI Scanner", f"Session {session_num}", "CAPTURE 450", "")
        captured["w450"] = capture_band_DI("w450", pins["w450"], deadline_t)

        oled_msg("MSI Scanner", f"Session {session_num}", "CAPTURE 660", "")
        captured["w660"] = capture_band_DI("w660", pins["w660"], deadline_t)

    except CaptureTimeout:
        all_leds_off()
        with open(base + "-TIMEOUT.txt", "w") as f:
            f.write(f"Capture timed out after {MAX_CAPTURE_SECONDS} seconds\n")
        oled_msg("MSI Scanner", f"Session {session_num}", "TIMEOUT", "", "")
        raise
    finally:
        all_leds_off()

    cap_s = time.monotonic() - t0
    oled_msg("MSI Scanner", f"Session {session_num}", f"CAP OK {cap_s:.1f}s", "PROCESSING...")
    process_and_save(session_dir, session_num, captured)

# ============================================================
# MAIN STATE MACHINE
# ============================================================
last_state = gpio.input(button_pin)
press_start = None
last_idle = 0.0

session_counter = 0
session_dir = None
stage = "IDLE"  # IDLE -> HAVE_STILL -> (CAL optional) -> SCAN

def main():
    global last_state, press_start, last_idle
    global session_counter, session_dir, stage

    oled_msg("MSI Scanner", "READY", "Press=STILL", "After: hold=CAL")

    try:
        while True:
            state = gpio.input(button_pin)
            now = time.monotonic()

            # idle UI refresh
            if now - last_idle > 0.6:
                cal = "Cal: OK" if have_calibration() else "Cal: none"
                oled_msg("MSI Scanner", f"Stage: {stage}", cal, "Press/hold btn")
                last_idle = now

            # press start
            if state == gpio.LOW and press_start is None:
                press_start = now

            # release edge
            if state == gpio.HIGH and last_state == gpio.LOW:
                held = 0.0 if press_start is None else (now - press_start)
                press_start = None

                # Stage logic
                if stage == "IDLE":
                    # First action MUST be still
                    session_counter += 1
                    session_dir = _new_session_dir(session_counter)
                    oled_msg("MSI Scanner", f"Session {session_counter}", "STILL...", "")
                    take_still_unmodified(session_dir, session_counter)
                    oled_msg("MSI Scanner", f"Session {session_counter}", "STILL saved ✅",
                             "Hold=CAL", "Press=SCAN")
                    stage = "HAVE_STILL"
                    continue

                if stage == "HAVE_STILL":
                    # After still: hold=cal, short=scan
                    if held >= LONG_PRESS_S:
                        # calibration
                        try:
                            capture_white_reference(session_dir, session_counter)
                            time.sleep(0.4)
                            oled_msg("MSI Scanner", f"Session {session_counter}",
                                     "READY", "Press=SCAN", "Hold=CAL (redo)")
                        except Exception:
                            oled_msg("CALIBRATION", "FAILED", "", "")
                            time.sleep(0.8)
                        continue
                    else:
                        # scan
                        try:
                            run_scan(session_dir, session_counter)
                        except Exception:
                            oled_msg("MSI Scanner", f"Session {session_counter}", "ERROR", "", "")
                            time.sleep(0.8)
                        finally:
                            session_dir = None
                            stage = "IDLE"
                            oled_msg("MSI Scanner", "READY", "Press=STILL", "After: hold=CAL")
                        continue

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
