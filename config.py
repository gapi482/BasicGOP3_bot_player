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