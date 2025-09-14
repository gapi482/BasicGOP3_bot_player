# Governor of Poker Bot

An automated poker bot for Governor of Poker 3 that uses computer vision and Monte Carlo simulation to play Texas Hold'em.

## Features

- **Card Detection**: Uses OpenCV for computer vision to detect cards on screen
- **Hand Evaluation**: Monte Carlo simulation to calculate win probabilities
- **Decision Making**: Basic poker strategy based on hand strength
- **Mouse Automation**: Simulates mouse clicks to play the game
- **Window Detection**: Automatically detects game window position
- **Calibration System**: Easy calibration for different screen setups

## Requirements

- Python 3.7+
- Windows OS (for window detection)
- Governor of Poker 3 running in windowed mode

## Installation

1. Install required dependencies:
```bash
pip install opencv-python numpy pyautogui pyscreenshot phevaluator pywin32

2. Run the bot:
-  python main.py