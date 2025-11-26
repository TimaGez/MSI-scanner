# MSI-scanner
This is the codebase for a custom multispectral imaging scanner I built.
### Hardware:
* 3x 18650 2500 mAh batteries to power, going through a 10A BMS protection board
* 10x Luminus SST-10 LEDs at wavelengths 450, 660, 730, 850, and 940 nm (2 of each)
* A 1.3" TFT touchscreen with capacitive touch sensor and ST7789 driver
* And controlling everything, a Raspberry Pi Zero 2W with the Raspberry Pi Camera Module v3 NoIR
* Paired, of course, with a USB Google Coral for AI capabilities 😈

### Software:
* Basic script to create a button on the screen which, when pressed, begins the scanning sequence
* (coming soon) A local ViT model to classify and predict skin cancer in real-time.
