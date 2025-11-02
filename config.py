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
    'enable_card_confirmation': True,  # Enable the card confirmation window
}

# Performance settings
PERFORMANCE = {
    'save_extracted_images': False,  # Save extracted card images to PNG (disable for faster execution)
    'save_screenshots': False,       # Save screenshots to PNG (disable for faster execution)
    'verbose_logging': False,        # Enable detailed logging (disable for faster execution)
}

# Poker card constants and default screen regions
CARD_RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
CARD_SUITS = ['h', 'd', 'c', 's']
CARD_EMOJIS = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}