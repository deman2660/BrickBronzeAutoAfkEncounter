# Roblox Brick Bronze │ Roria Conquest Pokémon Encounter Bot

A Python automation script for Roblox Pokémon games that walks your character in a small loop, detects wild encounters, reads the Pokémon name using OCR, compares it against a target list, and alerts you when a desired Pokémon appears.

The script automatically runs from encounters you do not want and pauses when a target Pokémon is found so you can catch it manually.

---

## Features

- Walks the character in a small circular pattern to trigger encounters
- Detects battle state using UI pixel sampling
- Reads Pokémon names using Tesseract OCR
- Uses fuzzy matching with stability checks to avoid OCR noise
- Automatically runs from non-target Pokémon
- Repeating audible alerts when a target Pokémon is found
- Pause, resume, stop, and target picker hotkeys
- Built-in Pokémon picker UI with live search
- Transparent, click-through overlay showing status and hotkeys

---

## Requirements

### Python
Python 3.10 or newer is recommended but 3.14 won't work. I personally choose 3.12.

### Python packages
Install all dependencies with pip:
```
pip install pywin32 mss numpy opencv-python pytesseract psutil pydirectinput pynput
```
---

## Tesseract OCR (Required)

This project requires **Tesseract OCR** to be installed.
pytesseract is only a wrapper and will not work without it.

### Download
Download the official Windows build from:
[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)

### Installation
During installation, make sure to enable:
Add Tesseract to system PATH

If you skipped this, manually add the following directory to your PATH environment variable:

C:\Program Files\Tesseract-OCR\

### Verify installation
Open a command prompt and run:
```
tesseract --version
```
If version information is printed, Tesseract is installed correctly.

---

## Files

ocr_pokemon.py  
Main script.

pokemon.txt  
A list of all Pokémon names, one per line.  

targets.txt  
Automatically created by the picker UI. Contains the Pokémon you want to hunt.

README.md  
Project documentation.

---

## Usage

1. Launch Roblox
2. Enter an area with wild Pokémon encounters
3. Run the script:
```
python ocr_pokemon.py
```
4. The Pokémon picker window will open
5. Search for and add one or more target Pokémon
6. Click Start
7. The script begins farming automatically

---

## Hotkeys

ESC  
Stops the script completely.

P  
Pause or unpause the script. No movement or clicks occur while paused.

R  
Resume farming after catching a target Pokémon.

O  
Open the Pokémon picker during runtime and change targets.

A translucent overlay at the top of the Roblox window shows current status and available hotkeys.

---

## How It Works

- The character walks in a small loop to trigger encounters.
- A specific UI pixel is sampled to detect when a battle starts.
- When a battle is detected:
  - OCR reads the Pokémon name from a fixed region of the screen.
  - The text is normalized and fuzzy-matched against pokemon.txt.
  - The name must be stable across multiple frames.
- If the Pokémon is a target:
  - The script alerts repeatedly.
  - It pauses until you catch the Pokémon and press R.
- If the Pokémon is not a target:
  - The script clicks Run.
  - It waits for the battle UI to disappear.
  - Farming resumes automatically.

---

## Known Limitations

- Roblox client must remain at 1280x720 resolution.
- UI scale must stay at default.
- Designed for Windows only.
- OCR accuracy depends on font clarity and lighting conditions.

---

## Disclaimer

This project is for educational and experimental purposes only.
Use responsibly and at your own risk.

---

## License

MIT License

