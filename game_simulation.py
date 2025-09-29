#!/usr/bin/env python3
"""
Game Simulation Module
"""
import random
from typing import List, Tuple
from phevaluator import evaluate_cards
import config

class GameSimulator:
    def __init__(self):
        """Initialize game simulator"""
        # Use unified constants
        self.suits = list(config.CARD_SUITS)
        self.ranks = list(config.CARD_RANKS)
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
        
        # Filter out None values and invalid cards
        hand = [card for card in hand if card is not None and len(card) == 2]
        table = [card for card in table if card is not None and len(card) == 2]
        
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
            if len(deck) >= 2:
                opponent_hands.append([deck.pop(), deck.pop()])
            else:
                break  # Not enough cards
        
        # Deal remaining table cards
        while len(table) < 5 and len(deck) > 0:
            table.append(deck.pop())
        
        try:
            # FIXED: Pass card strings directly to evaluate_cards
            # phevaluator can handle card strings like "As", "Kh", etc.
            all_my_cards = hand + table
            if len(all_my_cards) >= 2:
                my_hand_rank = evaluate_cards(*all_my_cards)
            else:
                print(f"Not enough cards for evaluation: {all_my_cards}")
                return 1  # Default to loss
            
            best_opponent_rank = float('inf')
            
            for opponent_hand in opponent_hands:
                opponent_cards = opponent_hand + table
                if len(opponent_cards) >= 2:  # Need at least 2 cards
                    try:
                        # FIXED: Pass card strings directly to evaluate_cards
                        opponent_rank = evaluate_cards(*opponent_cards)
                        best_opponent_rank = min(best_opponent_rank, opponent_rank)
                    except Exception as e:
                        print(f"Error evaluating opponent hand {opponent_cards}: {e}")
                        continue
            
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
        return [result / total for result in results] if total > 0 else [0.33, 0.33, 0.34]

    def make_decision(self, win_prob: float, pot_odds: float = 0.2) -> str:
        """Make decision based on win probability"""
        if win_prob < 0.3:  # Less than 30% chance to win
            return 'fold'
        elif win_prob < 0.6:  # 30-60% chance to win
            return 'check'
        else:  # More than 60% chance to win
            return 'bet'