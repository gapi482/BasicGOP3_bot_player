#!/usr/bin/env python3
"""
Configuration Settings
"""

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

# Poker card constants and default screen regions
CARD_RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
CARD_SUITS = ['h', 'd', 'c', 's']

# Sensible defaults; overridden by calibration when available
DEFAULT_GAME_WINDOW = {
    'left': 2,
    'top': 24,
    'width': 1920,
    'height': 1000,
}

DEFAULT_SCREEN_REGIONS = {
    'player_card1': (890, 620, 80, 100),
    'player_card2': (964, 620, 80, 100),
    'flop_cards': [(730, 390, 90, 110), (828, 390, 90, 110), (922, 390, 90, 110)],
    'turn_card': (1006, 390, 90, 110),
    'river_card': (1098, 390, 90, 110),
    'action_buttons': {
        'fold': (676, 948, 120, 60),
        'check': (1029, 948, 120, 60),
        'call': (1029, 948, 120, 60),
        'raise': (1412, 948, 120, 60),
        'bet': (1412, 948, 120, 60),
        'all_in': (1412, 948, 120, 60),
    }
}