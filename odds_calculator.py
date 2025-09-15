#!/usr/bin/env python3
"""
Professional Poker Odds Calculator using pbots_calc
"""
import requests
from typing import List, Tuple, Optional
from logger import Logger

class ProfessionalOddsCalculator:
    def __init__(self, server_url="http://localhost:5000/calculator"):
        """Initialize professional odds calculator"""
        self.server_url = server_url
        self.logger = Logger()
        
    def calculate_equity(self, hero_cards: List[str], board_cards: List[str], 
                        num_opponents: int = 1, num_simulations: int = 10000) -> Tuple[float, float, float]:
        """
        Calculate hand equity using professional poker calculator
        
        Args:
            hero_cards: List of hero's hole cards (e.g., ['As', 'Kh'])
            board_cards: List of community cards (e.g., ['2h', '5d', 'Jc'])
            num_opponents: Number of opponents
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Tuple of (win_probability, lose_probability, tie_probability)
        """
        try:
            # Format cards for pbots_calc
            hero_hand = ''.join(hero_cards)
            board = ''.join(board_cards) if board_cards else ''
            
            # Create opponent cards (unknown)
            opponents = ':XX' * num_opponents
            
            # Construct the query
            query = f"{hero_hand}{opponents}:{board}"
            
            # Send request to server
            response = requests.post(self.server_url, json={"command": query})
            
            if response.status_code == 200:
                result = response.json()
                if result:
                    equity = result[0][1]  # Extract equity value
                    # Convert to probabilities
                    win_prob = equity
                    lose_prob = 1 - equity - 0.02  # Small tie probability
                    tie_prob = 0.02
                    
                    return win_prob, lose_prob, tie_prob
                else:
                    self.logger.log("Empty response from odds calculator", level="WARNING")
                    return 0.33, 0.65, 0.02
            else:
                self.logger.log(f"Odds calculator error: {response.status_code}", level="ERROR")
                return self._fallback_calculation(hero_cards, board_cards)
                
        except Exception as e:
            self.logger.log(f"Odds calculation error: {e}", level="ERROR")
            return self._fallback_calculation(hero_cards, board_cards)
    
    def _fallback_calculation(self, hero_cards: List[str], board_cards: List[str]) -> Tuple[float, float, float]:
        """Fallback calculation using simplified method"""
        # Simple hand strength evaluation
        hero_cards_str = ''.join(hero_cards)
        board_cards_str = ''.join(board_cards)
        
        # Basic hand strength mapping (simplified)
        high_cards = ['A', 'K', 'Q', 'J']
        pairs = []
        
        # Check for pairs
        if hero_cards[0][0] == hero_cards[1][0]:
            pairs.append(hero_cards[0][0])
        
        # Calculate basic strength
        strength = 0.0
        
        # Add value for high cards
        for card in hero_cards:
            if card[0] in high_cards:
                strength += 0.1
        
        # Add value for pairs
        if pairs:
            strength += 0.2
        
        # Add value for suited cards
        if hero_cards[0][1] == hero_cards[1][1]:
            strength += 0.05
        
        # Ensure probability bounds
        strength = max(0.1, min(0.9, strength))
        
        return strength, 1 - strength - 0.02, 0.02