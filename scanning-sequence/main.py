import os
import time
import numpy as np
import RPi.GPIO as gpio

from datetime import date, datetime
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

# ---------------- GPIO ----------------
gpio.setmode(gpio.BCM)

pins = {
    "w730": 22,  # FIRST: 730 (dark)  -> old g730
    "w450": 17,  # SECOND: 450 (blue) -> old n450
    "w660": 23,  # THIRD: 660 (red)   -> old k850 pin, renamed
}

button_pin = 26

for p in pins.values():
    gpio.setup(p, gpio.OUT)
gpio.setup(button_pin, gpio.IN, pull_up_down=gpio.PUD_UP)

# ---------------- Camera ----------------
cam = Picamera2()

# Capture RAW Bayer so the ISP doesn't mess with spectral consistency.
cam.configure(cam.create_still_configuration(
    main={"format": "YUV420"},
    raw={"format": "SRGGB10"}
))
cam.start()

# Lock focus
cam.set_controls({
    "AfMode": 0,
    "LensPosition": 7.5
})

time.sleep(1.0)

# Kill auto + heavy processing (stable MSI)
cam.set_controls({
    "AeEnable": False,
    "AwbEnable": False,
    "Brightness": 0.0,
    "Contrast": 1.0,
    "Saturation": 0.0,
    "Sharpness": 0.0,
    "NoiseReductionMode": 0,
})

# ---------------- OLED ----------------
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial, width=128, height=64)
device.contrast(255)

WIDTH = device.width
HEIGHT = device.height
font = ImageFont.load_default()

# ---------------- Capture settings ----------------
# Tune these per LED. These are placeholders.
CAPTURE_SETTINGS = {
    "still": {"ExposureTime": 3500,  "AnalogueGain": 1.0, "LensPosition": 7.5},

    "w730":  {"ExposureTime": 20000, "AnalogueGain": 2.0, "LensPosition": 7.5},
    "w450":  {"ExposureTime": 20000, "AnalogueGain": 2.0, "LensPosition": 7.5},
    "w660":  {"ExposureTime": 20000, "AnalogueGain": 2.0, "LensPosition": 7.5},
}

CTRL_SETTLE = 0.15
LED_SETTLE  = 0.50
LED_OFF_GAP = 0.15

def apply_capture_settings(key: str) -> None:
    s = CAPTURE_SETTINGS[key]
    cam.set_controls({
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": int(s["ExposureTime"]),
        "AnalogueGain": float(s["AnalogueGain"]),
        "LensPosition": float(s["LensPosition"]),
    })

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _new_scan_dir(scan_num: int) -> str:
    timestamp = datetime.now().strftime("%H%M%S")
    folder = f"{date.today()}-scan{scan_num:03d}-{timestamp}"
    _ensure_dir(folder)
    return folder

def _split_rggb_planes(raw: np.ndarray):
    """SRGGB: R at [0,0], B at [1,1]"""
    raw = raw.astype(np.float32)
    R  = raw[0::2, 0::2]
    G1 = raw[0::2, 1::2]
    G2 = raw[1::2, 0::2]
    B  = raw[1::2, 1::2]
    return R, G1, G2, B

def _normalize_to_u8(img: np.ndarray) -> np.ndarray:
    """Per-band normalization for ML robustness."""
    x = img.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn)
    return (x * 255.0).clip(0, 255).astype(np.uint8)

def capture_band(key: str, outfile: str, plane: str) -> None:
    """
    plane:
      - 450nm -> usually strongest on B plane
      - 660/730 -> usually stronger on R plane
    """
    apply_capture_settings(key)
    time.sleep(CTRL_SETTLE)

    raw = cam.capture_array("raw")
    R, G1, G2, B = _split_rggb_planes(raw)

    if plane == "R":
        band = R
    elif plane == "B":
        band = B
    else:
        band = (G1 + G2) * 0.5

    Image.fromarray(_normalize_to_u8(band), mode="L").save(outfile)

def take_still(scan_dir: str, scan_num: int) -> None:
    apply_capture_settings("still")
    time.sleep(CTRL_SETTLE)
    cam.capture_file(os.path.join(scan_dir, f"{date.today()}-scan{scan_num:03d}-still.jpg"))

def sequence(scan_dir: str, scan_num: int) -> None:
    # ORDER updated:
    # 1) 730, 2) 450, 3) 660

    # ---- 730 ----
    gpio.output(pins["w730"], gpio.HIGH)
    time.sleep(LED_SETTLE)
    capture_band(
        "w730",
        os.path.join(scan_dir, f"{date.today()}-scan{scan_num:03d}-730nm.jpg"),
        plane="R"
    )
    gpio.output(pins["w730"], gpio.LOW)
    time.sleep(LED_OFF_GAP)

    # ---- 450 ----
    gpio.output(pins["w450"], gpio.HIGH)
    time.sleep(LED_SETTLE)
    capture_band(
        "w450",
        os.path.join(scan_dir, f"{date.today()}-scan{scan_num:03d}-450nm.jpg"),
        plane="B"
    )
    gpio.output(pins["w450"], gpio.LOW)
    time.sleep(LED_OFF_GAP)

    # ---- 660 ----
    gpio.output(pins["w660"], gpio.HIGH)
    time.sleep(LED_SETTLE)
    capture_band(
        "w660",
        os.path.join(scan_dir, f"{date.today()}-scan{scan_num:03d}-660nm.jpg"),
        plane="R"
    )
    gpio.output(pins["w660"], gpio.LOW)
    time.sleep(LED_OFF_GAP)

# ---------------- Main loop ----------------
last_state = gpio.input(button_pin)

scan_counter = 0
press_counter = 0
current_scan_dir = None

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

            # OLED status
            image = Image.new("1", (WIDTH, HEIGHT))
            draw = ImageDraw.Draw(image)

            draw.text((0, 0), "MSI Scanner", font=font, fill=255)
            draw.text((0, 16), f"Scan #: {scan_counter}", font=font, fill=255)

            next_phase = "STILL" if (press_counter % 2 == 0) else "SCAN"
            draw.text((0, 32), f"Next: {next_phase}", font=font, fill=255)

            btn_txt = "PRESSED" if state == gpio.LOW else "released"
            draw.text((0, 48), f"Button: {btn_txt}", font=font, fill=255)

            device.display(image)
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        device.display(Image.new("1", (WIDTH, HEIGHT)))
        cam.stop()
        gpio.cleanup()

if __name__ == "__main__":
    main()
