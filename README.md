# Governor of Poker Bot

An automated poker bot for Governor of Poker 3 that uses computer vision and Monte Carlo simulation to play Texas Hold'em. The bot features real-time card detection, an interactive confirmation window, and intelligent decision-making based on win probability calculations.

## Features

### Core Functionality
- **Advanced Card Detection**: Uses OpenCV template matching to detect player cards and community cards (flop, turn, river) with confidence thresholds
- **Interactive Card Confirmation Window**: GUI window (Tkinter) that allows you to:
  - View and correct detected cards before playing
  - Update card detection with fresh screenshots
  - Start new games with automatic card detection
  - Quick actions: Confirm & Play, Fold, Skip
- **Monte Carlo Simulation**: Calculates win probabilities using thousands of simulated hands (configurable sample size)
- **Intelligent Decision Making**: Makes poker decisions (fold/check/bet) based on configurable win probability thresholds
- **Automatic Window Detection**: Detects and focuses the Governor of Poker 3 game window
- **Screen Calibration System**: Easy calibration for different screen resolutions and setups

### Performance Optimizations
- **Configurable Performance Settings**: Disable image saving and verbose logging for faster execution
- **Optimized Screenshot Capture**: Direct memory conversion without disk I/O when saving is disabled
- **Efficient Template Matching**: Fast card recognition optimized for real-time gameplay

## Requirements

- **Python**: 3.7 or higher
- **Operating System**: Windows (required for window detection using `win32gui`)
- **Game**: Governor of Poker 3 running in windowed mode
- **Dependencies** (see Installation section)

## Installation

1. **Install Python dependencies:**
```bash
pip install opencv-python numpy pyautogui pyscreenshot phevaluator pywin32 pygetwindow pillow
```

Or install individually:
- `opencv-python` - Computer vision and image processing
- `numpy` - Numerical operations
- `pyautogui` - Mouse automation
- `pyscreenshot` - Screenshot capture
- `phevaluator` - Poker hand evaluation
- `pywin32` - Windows API access
- `pygetwindow` - Window detection
- `pillow` - Image processing

2. **Calibrate the bot:**
   - Ensure Governor of Poker 3 is running in windowed mode
   - Run the bot and use option 2 ("Test screen regions") to verify card detection
   - If needed, manually edit `screen_calibration.json` to adjust card positions

3. **Run the bot:**
```bash
python main.py
```

## Usage

### Main Menu Options

1. **Play single hand** - Captures cards, shows confirmation window, makes decision, and plays
2. **Test screen regions** - Visual test of all card and button regions
3. **Test card detection** - Test card detection accuracy on current game screen
4. **Preview game window** - Preview the captured game window
5. **Exit** - Exit the application

### Card Confirmation Window

When playing a hand, an interactive window appears with:

- **Player Cards Section**: Enter/correct your two hole cards (rank + suit buttons)
- **Table Cards Section**: Enter/correct community cards (flop, turn, river)
- **Action Buttons**:
  - **CONFIRM & PLAY**: Confirm cards and proceed with decision-making
  - **FOLD**: Immediately fold the hand
  - **SKIP**: Skip confirmation and use auto-detected cards
  - **UPDATE STATE**: Capture fresh screenshot and update only blank fields
  - **NEW GAME**: Clear all fields and capture fresh screenshot for new hand

### Configuration

Edit `config.py` to customize bot behavior:

```python
# Simulation settings
SIMULATION = {
    'monte_carlo_samples': 10000,  # Number of simulations per decision
    'default_opponents': 2,        # Number of opponents to simulate
}

# Bot behavior
BOT_BEHAVIOR = {
    'fold_threshold': 0.3,         # Fold if win probability < 30%
    'check_threshold': 0.6,        # Check if win probability 30-60%
    'bet_threshold': 0.6,          # Bet if win probability > 60%
    'delay_between_actions': 0.5,  # Delay after clicking buttons (seconds)
    'delay_between_hands': 3.0,    # Delay between hands (seconds)
    'enable_card_confirmation': True,  # Show confirmation window
}

# Performance settings (optimize for speed)
PERFORMANCE = {
    'save_extracted_images': False,  # Save card images to PNG files
    'save_screenshots': False,       # Save screenshots to PNG files
    'verbose_logging': False,        # Enable detailed console logging
}
```

**Performance Tip**: For fastest execution, keep all `PERFORMANCE` settings as `False`. Enable them only for debugging/calibration.

## How It Works

1. **Card Detection**: 
   - Captures screenshot of game window
   - Extracts card regions (player cards, flop, turn, river)
   - Matches extracted cards against template library using OpenCV
   - Applies confidence thresholds to filter false positives

2. **User Confirmation** (if enabled):
   - Shows detected cards in GUI window
   - User can correct any misdetections
   - Click "Confirm & Play" to proceed

3. **Decision Making**:
   - Runs Monte Carlo simulation with detected cards
   - Simulates thousands of possible opponent hands
   - Calculates win/loss/tie probabilities
   - Compares probabilities to thresholds and selects action

4. **Action Execution**:
   - Activates game window
   - Moves mouse to appropriate button (fold/check/bet)
   - Clicks button with slight randomization
   - Waits for game response

## File Structure

```
BasicGOP3_bot_player/
├── main.py                  # Main entry point
├── bot.py                   # Core bot logic and orchestration
├── card_detection.py        # Card detection and template matching
├── card_confirmation.py     # GUI confirmation window
├── game_simulation.py       # Monte Carlo simulation engine
├── utils.py                 # Window detection, screenshot capture
├── logger.py                # Logging functionality
├── config.py                # Configuration settings
├── screen_calibration.json  # Screen region calibration data
├── card_templates/          # Card template images (52 cards)
└── logs/                    # Application logs
```

## Troubleshooting

### Cards Not Detected Correctly
- Run "Test card detection" to see what the bot is detecting
- Check that `screen_calibration.json` has correct window coordinates
- Ensure game window is in windowed mode (not fullscreen)
- Verify card template images exist in `card_templates/` directory

### Bot Too Slow
- Set `PERFORMANCE` settings to `False` in `config.py`
- Reduce `monte_carlo_samples` (e.g., 5000 instead of 10000)
- Reduce `delay_between_actions` (if game allows faster clicking)

### Window Detection Issues
- Ensure Governor of Poker 3 window title contains "GOP3"
- Run as administrator if window detection fails
- Manually set window coordinates in `screen_calibration.json`

## Performance Notes

- Typical execution time per hand: **1-2 seconds**
  - Card detection: ~200-500ms
  - Monte Carlo simulation (5000-10000 samples): ~0.5-1.5s
  - Action execution: ~0.5-0.7s
- For best performance, disable image saving and verbose logging
- Reduce Monte Carlo samples if faster decisions are needed (trades accuracy for speed)

## License

This project is for educational purposes. Use responsibly and in accordance with game terms of service.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the bot's accuracy and performance.
