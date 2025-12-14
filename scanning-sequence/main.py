import os
import time
import json
import signal
import atexit
import traceback
import numpy as np
import RPi.GPIO as gpio

from datetime import date, datetime
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

# ============================================================
# BEHAVIOR
#   Press #1 -> STILL (unchanged JPG)
#   Press #2 -> CAL then SCAN (back-to-back, don't move)
# ============================================================

MAX_CAPTURE_SECONDS = 20.0  # CAPTURE budget only (processing can take longer)

# Hard-coded frames (kept tiny to guarantee no timeout)
CAL_ON_FRAMES  = {"w730": 1, "w450": 1, "w660": 1}
SCAN_ON_FRAMES = {"w730": 1, "w450": 1, "w660": 1}  # bump w450 to 2 only if you have time

# One global DARK capture reused across CAL+SCAN (major speedup)
DARK_FRAMES_GLOBAL = 1

# Crop to remove housing + reduce compute
USE_CENTER_CROP = True
CROP_SIZE = 700

# Output
SAVE_FLOAT16 = True
SAVE_PREVIEW_PNG = True
SAVE_PER_BAND_CORR = True
SAVE_ML_RATIOS = True

# Math stabilization
EPS = 1e-6
PEDESTAL_RAW = 20.0
CLIP_PCT = (1, 99)

# Timing
CTRL_SETTLE = 0.03
LED_SETTLE  = 0.06
LED_OFF_GAP = 0.01
INTER_FRAME_GAP = 0.001

RAW_BIT_DEPTH = 10
RAW_MAX = (1 << RAW_BIT_DEPTH) - 1  # 1023

# Button debounce / guard
DEBOUNCE_STABLE_S = 0.06
POST_PRESS_GUARD_S = 0.45

# Optional: stop double-presses from being counted as two presses
MIN_SECONDS_BETWEEN_PRESSES = 1.2

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

    # Tune later; these values won't affect capture-call overhead much
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
# OLED (boot-reliable)
# ============================================================
font = ImageFont.load_default()

def init_oled_with_retry(max_tries: int = 12, delay_s: float = 0.35):
    last_err = None
    for i in range(1, max_tries + 1):
        try:
            serial = i2c(port=1, address=0x3C)  # if yours is 0x3D, change here
            dev = ssd1306(serial, width=128, height=64)
            dev.contrast(255)
            img = Image.new("1", (dev.width, dev.height))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), "MSI Scanner", font=font, fill=255)
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
# FILE / IMAGE HELPERS
# ============================================================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _new_session_dir(session_num: int) -> str:
    ts = datetime.now().strftime("%H%M%S")
    folder = f"{date.today()}-session{session_num:03d}-{ts}"
    _ensure_dir(folder)
    return folder

def _save_npy(path: str, arr: np.ndarray):
    np.save(path, arr.astype(np.float16) if SAVE_FLOAT16 else arr.astype(np.float32))

def _save_preview_png(path: str, img_f32: np.ndarray):
    lo, hi = np.percentile(img_f32, (2, 98))
    if hi <= lo:
        vis = np.zeros_like(img_f32, dtype=np.uint8)
    else:
        vis = np.clip((img_f32 - lo) / (hi - lo), 0, 1)
        vis = (vis * 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(path)

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

# ============================================================
# BUTTON (debounced press event)
# ============================================================
_last_press_time = 0.0

def wait_for_debounced_press():
    global _last_press_time
    while True:
        while gpio.input(button_pin) == gpio.HIGH:
            time.sleep(0.005)

        t0 = time.monotonic()
        stable = True
        while time.monotonic() - t0 < DEBOUNCE_STABLE_S:
            if gpio.input(button_pin) != gpio.LOW:
                stable = False
                break
            time.sleep(0.005)
        if not stable:
            continue

        while gpio.input(button_pin) == gpio.LOW:
            time.sleep(0.005)

        time.sleep(POST_PRESS_GUARD_S)

        now = time.monotonic()
        if now - _last_press_time < MIN_SECONDS_BETWEEN_PRESSES:
            # ignore accidental double-presses
            continue

        _last_press_time = now
        return

# ============================================================
# CAPTURE CORE (fast request-based raw grab)
# ============================================================
class CaptureTimeout(Exception):
    pass

def _deadline_ok(deadline_t: float):
    if time.monotonic() > deadline_t:
        raise CaptureTimeout("Capture exceeded MAX_CAPTURE_SECONDS")

def _capture_raw_array_fast() -> np.ndarray:
    """
    Often faster/cleaner than capture_array("raw") and ensures request is released.
    """
    req = cam.capture_request()
    try:
        raw = req.make_array("raw")
    finally:
        req.release()
    return raw

def _capture_avg_intensity(n: int, deadline_t: float) -> (np.ndarray, list):
    acc = None
    times = []
    for _ in range(n):
        _deadline_ok(deadline_t)
        t0 = time.monotonic()
        raw = _capture_raw_array_fast()
        frame = _raw_to_intensity(raw)
        times.append(time.monotonic() - t0)
        acc = frame if acc is None else (acc + frame)
        time.sleep(INTER_FRAME_GAP)
    return acc / float(n), times

def capture_global_dark(deadline_t: float):
    all_leds_off()
    time.sleep(LED_OFF_GAP)
    D, times = _capture_avg_intensity(DARK_FRAMES_GLOBAL, deadline_t)
    return D, times

def capture_band_on(band: str, led_pin: int, frames: int, deadline_t: float):
    ctrl = apply_capture_settings(band)
    time.sleep(CTRL_SETTLE)
    _deadline_ok(deadline_t)

    gpio.output(led_pin, gpio.HIGH)
    time.sleep(LED_SETTLE)
    I, times = _capture_avg_intensity(frames, deadline_t)
    gpio.output(led_pin, gpio.LOW)
    time.sleep(LED_OFF_GAP)

    return I, ctrl, times

# ============================================================
# STILL (UNCHANGED)
# ============================================================
def take_still_unmodified(session_dir: str, session_num: int) -> str:
    apply_capture_settings("still")
    time.sleep(CTRL_SETTLE)
    path = os.path.join(session_dir, f"{date.today()}-session{session_num:03d}-still.jpg")
    cam.capture_file(path)  # untouched
    return path

# ============================================================
# CAL + SCAN (global dark frame -> big speedup)
# ============================================================
def do_cal_and_scan(session_dir: str, session_num: int):
    base = os.path.join(session_dir, f"{date.today()}-session{session_num:03d}")
    deadline_t = time.monotonic() + MAX_CAPTURE_SECONDS

    meta = {
        "session_num": session_num,
        "timestamp": datetime.now().isoformat(),
        "crop": {"enabled": USE_CENTER_CROP, "size": CROP_SIZE},
        "cal_on_frames": CAL_ON_FRAMES,
        "scan_on_frames": SCAN_ON_FRAMES,
        "dark_frames_global": DARK_FRAMES_GLOBAL,
        "pedestal_raw": PEDESTAL_RAW,
        "clip_percentiles": CLIP_PCT,
        "bands": {},
    }

    # ---------- DARK once ----------
    oled_msg("CAL+SCAN", "DO NOT MOVE", "DARK frame...", "")
    D, dark_times = capture_global_dark(deadline_t)
    meta["dark_frame_seconds"] = dark_times

    # ---------- CAL ----------
    oled_msg("CAL", "Capturing...", "730/450/660", "")
    flats = {}

    for band, nm in [("w730","730"), ("w450","450"), ("w660","660")]:
        oled_msg("CAL", f"ON {nm}nm", "capturing...", "")
        I, ctrl, on_times = capture_band_on(band, pins[band], CAL_ON_FRAMES[band], deadline_t)

        W = (I - D) + PEDESTAL_RAW
        W = np.clip(W, 0, RAW_MAX).astype(np.float32)
        W = _clip_percentile(W, CLIP_PCT)
        flats[band] = W

        meta["bands"][band] = {
            "name": f"{nm}nm",
            "camera_controls": ctrl,
            "cal_on_frame_seconds": on_times,
        }

        out = f"{base}-flat-{nm}nm"
        _save_npy(out + ".npy", W)
        if SAVE_PREVIEW_PNG:
            _save_preview_png(out + "-preview.png", W)

    # ---------- SCAN ----------
    oled_msg("SCAN", "DO NOT MOVE", "Capturing...", "")
    corr = {}

    for band, nm in [("w730","730"), ("w450","450"), ("w660","660")]:
        oled_msg("SCAN", f"ON {nm}nm", "capturing...", "")
        I, ctrl, on_times = capture_band_on(band, pins[band], SCAN_ON_FRAMES[band], deadline_t)

        X = (I - D) + PEDESTAL_RAW
        X = np.clip(X, 0, RAW_MAX).astype(np.float32)

        # Flat-field normalize with session-local flat
        X = X / (flats[band] + EPS)
        X = _clip_percentile(X, CLIP_PCT)
        corr[band] = X

        meta["bands"][band]["scan_on_frame_seconds"] = on_times

        if SAVE_PER_BAND_CORR:
            out = f"{base}-{nm}nm-corr"
            _save_npy(out + ".npy", X)
            if SAVE_PREVIEW_PNG:
                _save_preview_png(out + "-preview.png", X)

    if SAVE_ML_RATIOS:
        r_660_730 = _clip_percentile(corr["w660"] / (corr["w730"] + EPS), CLIP_PCT)
        r_450_660 = _clip_percentile(corr["w450"] / (corr["w660"] + EPS), CLIP_PCT)
        r_450_730 = _clip_percentile(corr["w450"] / (corr["w730"] + EPS), CLIP_PCT)

        tensor = np.stack([r_660_730, r_450_660, r_450_730], axis=-1).astype(np.float32)
        out_ml = f"{base}-ML-ratios"
        _save_npy(out_ml + ".npy", tensor)

        if SAVE_PREVIEW_PNG:
            _save_preview_png(out_ml + "-ch0-660over730.png", tensor[..., 0])
            _save_preview_png(out_ml + "-ch1-450over660.png", tensor[..., 1])
            _save_preview_png(out_ml + "-ch2-450over730.png", tensor[..., 2])

        meta["ml_tensor"] = {
            "file": os.path.basename(out_ml + ".npy"),
            "channels": ["660/730", "450/660", "450/730"],
        }

    with open(base + "-meta.json", "w") as f:
        json.dump(meta, f, indent=2)

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    session_counter = 0
    stage = "WAIT_STILL"   # WAIT_STILL -> WAIT_CALSCAN
    session_dir = None
    session_num = None

    oled_msg("MSI Scanner", "READY", "Press: STILL", "", "")

    try:
        while True:
            wait_for_debounced_press()

            if stage == "WAIT_STILL":
                session_counter += 1
                session_num = session_counter
                session_dir = _new_session_dir(session_num)

                oled_msg("MSI Scanner", f"Session {session_num}", "Taking STILL...", "")
                take_still_unmodified(session_dir, session_num)

                oled_msg("MSI Scanner",
                         f"Session {session_num}",
                         "STILL saved ✅",
                         "Press: CAL+SCAN")
                stage = "WAIT_CALSCAN"

            else:
                base = os.path.join(session_dir, f"{date.today()}-session{session_num:03d}")
                oled_msg("CAL+SCAN", f"Session {session_num}", "DO NOT MOVE", "")

                try:
                    do_cal_and_scan(session_dir, session_num)
                    oled_msg("DONE ✅", f"Session {session_num}", "Saved", "")
                except CaptureTimeout:
                    all_leds_off()
                    with open(base + "-TIMEOUT.txt", "w") as f:
                        f.write(f"Capture timed out after {MAX_CAPTURE_SECONDS} seconds\n")
                    oled_msg("TIMEOUT", f"{MAX_CAPTURE_SECONDS:.0f}s cap", "reduce frames", "")
                except Exception:
                    all_leds_off()
                    err_path = base + "-ERROR.txt"
                    with open(err_path, "w") as f:
                        f.write(traceback.format_exc())
                    oled_msg("ERROR", "Saved ERROR.txt", "", "")
                finally:
                    all_leds_off()

                # reset for next session
                stage = "WAIT_STILL"
                session_dir = None
                session_num = None
                oled_msg("MSI Scanner", "READY", "Press: STILL", "")

    finally:
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
        gpio.cleanup()

if __name__ == "__main__":
    main()
