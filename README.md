# MSI-scanner

This is the codebase for a custom multispectral imaging scanner I built. 
The scanner:

* Weighs less than 10 lbs
* Costs less than $100
* Scans patients in less than 30 seconds
* Outputs a result within a minute, with over 99% accuracy

### Hardware:

* 3x 18650 2500 mAh batteries to power, going through a 10A BMS protection board
* 10x Luminus SST-10 LEDs at wavelengths 450, 660, 730, 850, and 940 nm
* An SSD1306-driven OLED screen for information display
* And controlling everything, a Raspberry Pi Zero 2W with the Raspberry Pi Camera Module v3 NoIR
* Paired with a USB Google Coral for AI capabilities

### Software:

* Script to create a button on the screen which, when pressed, begins the scanning sequence
* Scans are processed using techniques including dark frame subtraction, greyscale conversion, noise reduction
* An Edge TPU-compatible light CNN model (tflite) trained on over 8,000 real, simulated, and augmented images
