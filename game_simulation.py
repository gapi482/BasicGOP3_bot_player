#!/usr/bin/env python3
"""
Game Simulation Module
"""

import random
from typing import List, Tuple
from phevaluator import evaluate_cards

class GameSimulator:
    def __init__(self):
        """Initialize game simulator"""
        self.suits = ['d', 's', 'c', 'h']
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        self.cards = [r + s for r in self.ranks for s in self.suits]
    
    def card_to_values(self, card: str) -> Tuple[int, int]:
        """Convert card string to rank and suit values for phevaluator"""
        if not card or len(card) < 2:
            return 14, 1  # Default to Ace of Spades
        
        rank_char = card[0]
        suit_char = card[1]
        
        rank_map = {
            'A': 14,  # Ace
            'K': 13,  # King
            'Q': 12,  # Queen
            'J': 11,  # Jack
            'T': 10   # Ten
        }
        
        # Get rank value
        if rank_char in rank_map:
            rank = rank_map[rank_char]
        else:
            try:
                rank = int(rank_char)
            except ValueError:
                rank = 14
                print(f"Warning: Invalid rank '{rank_char}' in card '{card}', defaulting to Ace")
        
        # Suit mapping
        suit_map = {'s': 1, 'h': 2, 'd': 3, 'c': 4}
        suit = suit_map.get(suit_char, 1)  # Default to spades
        
        return rank, suit
    
    def simulate_hand(self, hand: List[str], table: List[str], players: int = 2) -> int:
        """Simulate one hand of poker"""
        # Validate inputs
        if not hand or len(hand) < 2:
            print(f"Invalid hand: {hand}")
            return 1  # Default to loss
        
        # Filter out None values
        hand = [card for card in hand if card is not None]
        table = [card for card in table if card is not None]
        
        if len(hand) < 2:
            print(f"Invalid hand after filtering: {hand}")
            return 1  # Default to loss
        
        deck = self.cards.copy()
        
        # Remove known cards from deck
        known_cards = hand + table
        deck = [card for card in deck if card not in known_cards]
        
        # Shuffle remaining deck
        random.shuffle(deck)
        
        # Deal cards to opponents
        opponent_hands = []
        for _ in range(players - 1):
            opponent_hands.append([deck.pop(), deck.pop()])
        
        # Deal remaining community cards
        while len(table) < 5:
            if deck:  # Make sure we have cards left
                table.append(deck.pop())
            else:
                break  # No more cards to deal
        
        # Evaluate all hands - FIXED: Properly handle card conversion
        try:
            # Convert cards to the format expected by phevaluator
            my_hand_cards = []
            for card in hand:
                if card and len(card) == 2:  # Valid card string
                    rank, suit = self.card_to_values(card)
                    my_hand_cards.append((rank, suit))
            
            if len(my_hand_cards) < 2:
                print(f"Invalid hand cards after conversion: {my_hand_cards}")
                return 1  # Default to loss
            
            # Evaluate my hand
            my_hand_rank = evaluate_cards(*my_hand_cards)
            
            best_opponent_rank = float('inf')
            for opponent_hand in opponent_hands:
                opponent_cards = opponent_hand + table
                
                # Convert opponent cards
                opponent_card_values = []
                for card in opponent_cards:
                    if card and len(card) == 2:
                        rank, suit = self.card_to_values(card)
                        opponent_card_values.append((rank, suit))
                
                if len(opponent_card_values) >= 2:  # Need at least 2 cards
                    try:
                        opponent_rank = evaluate_cards(*opponent_card_values)
                        best_opponent_rank = min(best_opponent_rank, opponent_rank)
                    except Exception as e:
                        print(f"Error evaluating opponent hand: {e}")
            
            # Return result: 0 = win, 1 = lose, 2 = tie
            if my_hand_rank < best_opponent_rank:
                return 0  # Win
            elif my_hand_rank == best_opponent_rank:
                return 2  # Tie
            else:
                return 1  # Lose
                
        except Exception as e:
            print(f"Error in hand evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 1  # Default to loss
        
    def monte_carlo_simulation(self, hand: List[str], table: List[str], players: int = 2, samples: int = 10000) -> List[float]:
        """Run Monte Carlo simulation to calculate win probabilities"""
        results = [0, 0, 0]  # [wins, losses, ties]
            
        for _ in range(samples):
            outcome = self.simulate_hand(hand, table, players)
            results[outcome] += 1
            
        # Convert to percentages
        total = sum(results)
        return [result / total for result in total] if total > 0 else [0.33, 0.33, 0.34]
        
    def make_decision(self, win_prob: float, pot_odds: float = 0.2) -> str:
        """Make decision based on win probability"""
        if win_prob < 0.3:  # Less than 30% chance to win
            return 'fold'
        elif win_prob < 0.6:  # 30-60% chance to win
            return 'check'
        else:  # More than 60% chance to win
            return 'bet'