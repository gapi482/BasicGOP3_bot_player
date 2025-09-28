#!/usr/bin/env python3
"""
Configuration Settings
"""

# Screen settings
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# System bars (typical Windows setup)
SYSTEM_BARS = {
    'top_bar_height': 40,    # Windows title bar
    'bottom_bar_height': 40,  # Windows taskbar
    'side_margin': 0,         # No side margins
}

# Card detection settings
CARD_DETECTION = {
    'confidence_threshold': 0.5,
    'template_matching_threshold': 0.8,
    'color_detection_enabled': True,
    'contour_detection_enabled': True,
}

# Game simulation settings
SIMULATION = {
    'monte_carlo_samples': 10000,
    'default_opponents': 2,
}

# Bot behavior settings
BOT_BEHAVIOR = {
    'fold_threshold': 0.3,    # Fold if win probability < 30%
    'check_threshold': 0.6,   # Check if win probability 30-60%
    'bet_threshold': 0.6,     # Bet if win probability > 60%
    'delay_between_actions': 0.5,
    'delay_between_hands': 3.0,
    'enable_card_confirmation': True,  # Enable card confirmation window
}

# File paths
FILE_PATHS = {
    'calibration_file': 'screen_calibration.json',
    'template_directory': 'card_templates',
    'debug_directory': 'debug',
}

# Debug settings
DEBUG = {
    'enabled': True,
    'save_debug_images': True,
    'verbose_output': True,
}

# Poker card constants and default screen regions
CARD_RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
CARD_SUITS = ['h', 'd', 'c', 's']

# Sensible defaults; overridden by calibration when available
DEFAULT_GAME_WINDOW = {
    'left': 0,
    'top': 40,
    'width': 1920,
    'height': 1000,
}

DEFAULT_SCREEN_REGIONS = {
    'player_card1': (899, 628, 60, 80),
    'player_card2': (973, 628, 60, 80),
    'flop_cards': [(540, 450, 70, 90), (620, 450, 70, 90), (700, 450, 70, 90)],
    'turn_card': (780, 450, 70, 90),
    'river_card': (860, 450, 70, 90),
    'action_buttons': {
        'fold': (1230, 700, 100, 40),
        'check': (1230, 700, 100, 40),
        'call': (1230, 700, 100, 40),
        'raise': (1350, 700, 100, 40),
        'all_in': (1470, 700, 100, 40),
    }
}