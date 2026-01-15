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

MAX_CAPTURE_SECONDS = 60.0 

RAW_BIT_DEPTH = 10
RAW_MAX = (1 << RAW_BIT_DEPTH) - 1

# subtract  global dark frame, add tiny pedestal to avoid division by zero
EPS = 1e-6
PEDESTAL = 20.0

# if too many pixels near 0 or max, refuse to save
CLIP_HIGH_THRESH = int(0.98 * RAW_MAX)   # "near saturation"
CLIP_LOW_THRESH  = int(0.01 * RAW_MAX)   # "near black"
MAX_CLIP_FRAC = 0.003                    # max allowed fraction clipped (0.3%)

USE_CENTER_CROP = True
CROP_SIZE = 700

USE_CIRCULAR_MASK = True
CIRCULAR_MASK_FRAC = 0.47

SAVE_PREVIEW_PNG = True
PREVIEW_CLIP_PCT = (2, 98)

SAVE_FLOAT16 = True  # smaller files
SAVE_PER_BAND_RAW = True
SAVE_PER_BAND_CORR = True
SAVE_ML_TENSOR = True

# timing
CTRL_SETTLE = 0.03
LED_SETTLE  = 0.06
LED_OFF_GAP = 0.02
POST_PRESS_GUARD_S = 0.45
DEBOUNCE_STABLE_S = 0.06
MIN_SECONDS_BETWEEN_PRESSES = 1.2

gpio.setmode(gpio.BCM)

pins = {
    "w730": 17,
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


cam = Picamera2()
cam.configure(cam.create_still_configuration(
    main={"format": "YUV420"},
    raw={"format": "SRGGB10"}
))
cam.start()
time.sleep(0.6)

# disable auto everything
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

# per-band settings
CAPTURE_SETTINGS = {
    "still": {"ExposureTime": 2500,  "AnalogueGain": 1.0, "LensPosition": 7.5},
    "w730":  {"ExposureTime": 23000, "AnalogueGain": 1.5, "LensPosition": 7.5},
    "w660":  {"ExposureTime": 30000, "AnalogueGain": 1.5, "LensPosition": 7.5},
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

def capture_raw_array() -> np.ndarray:
    req = cam.capture_request()
    try:
        raw = req.make_array("raw")
    finally:
        req.release()
    return raw

# OLED screen
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

# helpers for scan processing
class CaptureAbort(Exception):
    pass

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _new_session_dir(session_num: int) -> str:
    ts = datetime.now().strftime("%H%M%S")
    folder = f"{date.today()}-session{session_num:03d}-{ts}"
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

def _make_circular_mask(h: int, w: int, frac: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = frac * min(h, w)
    mask = ((yy - cy)**2 + (xx - cx)**2) <= (r**2)
    return mask.astype(np.float32)

def _split_rggb_planes(raw: np.ndarray):
    raw = raw.astype(np.float32)
    R  = raw[0::2, 0::2]
    G1 = raw[0::2, 1::2]
    G2 = raw[1::2, 0::2]
    B  = raw[1::2, 1::2]
    G = (G1 + G2) * 0.5
    return R, G, B

def raw_to_intensity_masked(raw: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Convert SRGGB10 raw;
    crop + circular mask
    """
    R, G, B = _split_rggb_planes(raw)
    I = (R + 2.0*G + B) * 0.25
    I = _center_crop(I, CROP_SIZE).astype(np.float32)

    if USE_CIRCULAR_MASK:
        m = _make_circular_mask(I.shape[0], I.shape[1], CIRCULAR_MASK_FRAC)
        I = I * m
        return I, m
    else:
        m = np.ones_like(I, dtype=np.float32)
        return I, m

def _save_npy(path: str, arr: np.ndarray):
    np.save(path, arr.astype(np.float16) if SAVE_FLOAT16 else arr.astype(np.float32))

def _save_preview_png(path: str, img_f32: np.ndarray):
    lo, hi = np.percentile(img_f32, PREVIEW_CLIP_PCT)
    if hi <= lo:
        vis = np.zeros_like(img_f32, dtype=np.uint8)
    else:
        vis = np.clip((img_f32 - lo) / (hi - lo), 0, 1)
        vis = (vis * 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(path)

def check_clipping(x: np.ndarray, mask: np.ndarray, label: str):
#     """
#     makes sure masked ROI is not saturated or crushed
#     if so, abort scan to ensure only good data
#     """
    roi = x[mask > 0.5]
#     if roi.size < 1000:
#         raise CaptureAbort(f"{label}: ROI too small (mask/crop wrong)")

    hi_frac = float(np.mean(roi >= CLIP_HIGH_THRESH))
    lo_frac = float(np.mean(roi <= CLIP_LOW_THRESH))

#     if hi_frac > MAX_CLIP_FRAC or lo_frac > MAX_CLIP_FRAC:
#         raise CaptureAbort(
#             f"{label}: clipping too high "
#             f"(hi={hi_frac*100:.2f}%, lo={lo_frac*100:.2f}%). "
#             f"Lower/raise ExposureTime in CAPTURE_SETTINGS."
#         )
    return {"hi_clip_frac": hi_frac, "lo_clip_frac": lo_frac}

# button
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
            continue
        _last_press_time = now
        return

# still image
def take_still_unmodified(session_dir: str, session_num: int) -> str:
    apply_capture_settings("still")
    time.sleep(CTRL_SETTLE)
    path = os.path.join(session_dir, f"{date.today()}-session{session_num:03d}-still.jpg")
    cam.capture_file(path)  # do not touch
    return path

# scan capture (1 per led)
def capture_dark_single(deadline_t: float):
    if time.monotonic() > deadline_t:
        raise CaptureAbort("Time budget exceeded before DARK")
    all_leds_off()
    time.sleep(LED_OFF_GAP)
    raw = capture_raw_array()
    I, mask = raw_to_intensity_masked(raw)
    return I, mask

def capture_band_single(band: str, led_pin: int, deadline_t: float):
    if time.monotonic() > deadline_t:
        raise CaptureAbort("Time budget exceeded before band capture")
    ctrl = apply_capture_settings(band)
    time.sleep(CTRL_SETTLE)

    gpio.output(led_pin, gpio.HIGH)
    time.sleep(LED_SETTLE)

    raw = capture_raw_array()
    I, mask = raw_to_intensity_masked(raw)

    gpio.output(led_pin, gpio.LOW)
    time.sleep(LED_OFF_GAP)
    return I, mask, ctrl

# calibration + scan sequence
def do_cal_and_scan(session_dir: str, session_num: int):
    base = os.path.join(session_dir, f"{date.today()}-session{session_num:03d}")
    deadline_t = time.monotonic() + MAX_CAPTURE_SECONDS

    meta = {
        "session_num": session_num,
        "timestamp": datetime.now().isoformat(),
        "bands_used": ["730nm", "660nm"],
        "single_frame_per_state": True,
        "crop": {"enabled": USE_CENTER_CROP, "size": CROP_SIZE},
        "circular_mask": {"enabled": USE_CIRCULAR_MASK, "frac": CIRCULAR_MASK_FRAC},
        "clip_rules": {
            "hi_thresh": CLIP_HIGH_THRESH,
            "lo_thresh": CLIP_LOW_THRESH,
            "max_clip_frac": MAX_CLIP_FRAC
        },
        "camera_settings": {
            "w730": CAPTURE_SETTINGS["w730"],
            "w660": CAPTURE_SETTINGS["w660"],
        },
        "bands": {}
    }

    # dark frame
    oled_msg("CAL+SCAN", "DO NOT MOVE", "DARK (1 frame)", "")
    D, mask = capture_dark_single(deadline_t)
    meta["dark"] = {
        "shape": list(D.shape),
        "mean": float(np.mean(D[mask > 0.5])),
    }

    # calibration
    flats = {}

    for band, nm in [("w730", "730"), ("w660", "660")]:
        oled_msg("CAL", f"{nm}nm ON", "1 frame", "")
        I_on, mask2, ctrl = capture_band_single(band, pins[band], deadline_t)

        # ensure mask is regular shape
        if mask2.shape != mask.shape:
            raise CaptureAbort("Mask shape mismatch (crop/mask instability)")

        # dark subtracted, pedestal added
        W = (I_on - D) + PEDESTAL
        W = np.clip(W, 0, RAW_MAX).astype(np.float32)

        clip_stats = check_clipping(W, mask, f"CAL {nm}nm")
        flats[band] = W

        meta["bands"][band] = {
            "name": f"{nm}nm",
            "camera_controls": ctrl,
            "cal_clip": clip_stats,
            "cal_mean": float(np.mean(W[mask > 0.5])),
            "cal_max": float(np.max(W[mask > 0.5])),
        }

        if SAVE_PER_BAND_RAW:
            out = f"{base}-flat-{nm}nm"
            _save_npy(out + ".npy", W)
            if SAVE_PREVIEW_PNG:
                _save_preview_png(out + "-preview.png", W)

    # actual scan
    corr = {}
    rawscan = {}

    for band, nm in [("w730", "730"), ("w660", "660")]:
        oled_msg("SCAN", f"{nm}nm ON", "1 frame", "")
        I_on, mask2, ctrl = capture_band_single(band, pins[band], deadline_t)

        if mask2.shape != mask.shape:
            raise CaptureAbort("Mask shape mismatch (crop/mask instability)")

        X = (I_on - D) + PEDESTAL
        X = np.clip(X, 0, RAW_MAX).astype(np.float32)

        clip_stats = check_clipping(X, mask, f"SCAN {nm}nm")
        rawscan[band] = X

        # flat-field correction
        Xcorr = X / (flats[band] + EPS)
        corr[band] = Xcorr.astype(np.float32)

        meta["bands"][band]["scan_clip"] = clip_stats
        meta["bands"][band]["scan_mean"] = float(np.mean(X[mask > 0.5]))
        meta["bands"][band]["scan_max"] = float(np.max(X[mask > 0.5]))

        # save raw + corrected
        if SAVE_PER_BAND_RAW:
            out_raw = f"{base}-{nm}nm-raw"
            _save_npy(out_raw + ".npy", X)
            if SAVE_PREVIEW_PNG:
                _save_preview_png(out_raw + "-preview.png", X)

        if SAVE_PER_BAND_CORR:
            out_corr = f"{base}-{nm}nm-corr"
            _save_npy(out_corr + ".npy", Xcorr)
            if SAVE_PREVIEW_PNG:
                _save_preview_png(out_corr + "-preview.png", Xcorr)

    # create + save ml tensors
    if SAVE_ML_TENSOR:
        ratio = corr["w660"] / (corr["w730"] + EPS)
        tensor = np.stack([corr["w660"], corr["w730"], ratio], axis=-1).astype(np.float32)

        out_ml = f"{base}-ML-tensor"
        _save_npy(out_ml + ".npy", tensor)

        if SAVE_PREVIEW_PNG:
            _save_preview_png(out_ml + "-ch0-660.png", tensor[..., 0])
            _save_preview_png(out_ml + "-ch1-730.png", tensor[..., 1])
            _save_preview_png(out_ml + "-ch2-660over730.png", tensor[..., 2])

        meta["ml_tensor"] = {
            "file": os.path.basename(out_ml + ".npy"),
            "channels": ["660_corr", "730_corr", "660/730"],
            "shape": list(tensor.shape),
        }

    # save meta for diagnostics
    with open(base + "-meta.json", "w") as f:
        json.dump(meta, f, indent=2)

# main loop
def main():
    session_counter = 0
    stage = "WAIT_STILL"
    session_dir = None
    session_num = None

    oled_msg("MSI Scanner", "READY", "Press: STILL", "")

    try:
        while True:
            wait_for_debounced_press()

            if stage == "WAIT_STILL":
                session_counter += 1
                session_num = session_counter
                session_dir = _new_session_dir(session_num)

                oled_msg("MSI Scanner", f"Session {session_num}", "Taking STILL", "")
                take_still_unmodified(session_dir, session_num)

                oled_msg("MSI Scanner", f"Session {session_num}", "STILL saved", "Press: CAL+SCAN")
                stage = "WAIT_CALSCAN"

            else:
                base = os.path.join(session_dir, f"{date.today()}-session{session_num:03d}")
                oled_msg("CAL+SCAN", f"Session {session_num}", "DO NOT MOVE", "")

                try:
                    do_cal_and_scan(session_dir, session_num)
                    oled_msg("DONE", f"Session {session_num}", "Saved (VALID)", "")
                except CaptureAbort as e:
                    all_leds_off()
                    warn_path = base + "-INVALID.txt"
                    with open(warn_path, "w") as f:
                        f.write(str(e) + "\n")
                    oled_msg("INVALID DATA", "Not saved", "Check exposure", "")
                except Exception:
                    all_leds_off()
                    err_path = base + "-ERROR.txt"
                    with open(err_path, "w") as f:
                        f.write(traceback.format_exc())
                    oled_msg("ERROR", "Saved ERROR.txt", "", "")
                finally:
                    all_leds_off()

                # reset
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
