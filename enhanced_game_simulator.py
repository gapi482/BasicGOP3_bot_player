#!/usr/bin/env python3
"""
Enhanced Game Simulation Module
"""
from advanced_card_detector import AdvancedCardDetector
from odds_calculator import ProfessionalOddsCalculator
from logger import Logger

class EnhancedGameSimulator:
    def __init__(self):
        """Initialize enhanced game simulator"""
        self.card_detector = AdvancedCardDetector()
        self.odds_calculator = ProfessionalOddsCalculator()
        self.logger = Logger()
        
    def simulate_hand(self, img, screen_regions, game_window, num_opponents: int = 1):
        """Simulate a hand with enhanced detection and odds calculation"""
        try:
            # Detect cards using advanced methods
            player_cards = self.card_detector.get_player_cards(img, screen_regions, game_window)
            community_cards = self.card_detector.get_community_cards(img, screen_regions, game_window)
            
            self.logger.log(f"Detected player cards: {player_cards}")
            self.logger.log(f"Detected community cards: {community_cards}")
            
            # Validate cards
            if len(player_cards) < 2:
                self.logger.log("Insufficient player cards detected", level="WARNING")
                return None, None, None
            
            # Calculate professional odds
            win_prob, lose_prob, tie_prob = self.odds_calculator.calculate_equity(
                player_cards, community_cards, num_opponents
            )
            
            self.logger.log(f"Win probability: {win_prob:.2%}")
            self.logger.log(f"Lose probability: {lose_prob:.2%}")
            self.logger.log(f"Tie probability: {tie_prob:.2%}")
            
            return win_prob, lose_prob, tie_prob
            
        except Exception as e:
            self.logger.log_error(f"Hand simulation error: {e}", e)
            return None, None, None
    
    def make_decision(self, win_prob: float, pot_odds: float = 0.2) -> str:
        """Make enhanced decision based on probabilities"""
        if win_prob < 0.3:
            return 'fold'
        elif win_prob < 0.5:
            return 'check'
        elif win_prob < 0.7:
            return 'call'
        else:
            return 'bet'