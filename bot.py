#!/usr/bin/env python3
"""
Main Bot Class for Governor of Poker
"""
import random
import cv2
import numpy as np
import pyautogui
import time
import json
import os
import traceback
from logger import Logger
from card_detection import CardDetector
from game_simulation import GameSimulator
from utils import GameWindowCapture, WindowDetector, ScreenshotManager

class GovernorOfPokerBot:
    def __init__(self, calibration_file='screen_calibration.json', logger=None):
        """Initialize the bot with calibration data"""
        self.calibration_file = calibration_file
        self.calibration_data = None
        
        # Initialize logger
        self.logger = Logger()
        self.logger.log("Initializing Governor of Poker Bot")
        
        # Load calibration data
        if os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            self._setup_from_calibration()
            self.logger.log("Calibration data loaded successfully")
        else:
            self._setup_defaults()
            self.logger.log("No calibration file found, using defaults", level="WARNING")
        
        # Initialize components
        self.window_detector = WindowDetector()
        self.screenshot_manager = ScreenshotManager(self.game_window)
        self.card_detector = CardDetector()
        self.game_simulator = GameSimulator()
        
        # Initialize simple odds calculator (fallback)
        self.odds_calculator = SimpleOddsCalculator()
        
        # Safety settings
        pyautogui.PAUSE = 0.5
        pyautogui.FAILSAFE = True
        self.logger.log("Bot initialized successfully")

    def _setup_from_calibration(self):
        """Setup from calibration data"""
        self.game_window = self.calibration_data['game_window']
        self.screen_regions = self.calibration_data['screen_regions']

    def _setup_defaults(self):
        """Setup default values"""
        self.game_window = {
            'left': 0,
            'top': 40,
            'width': 1920,
            'height': 1000
        }
        
        self.screen_regions = {
            'player_card1': (899, 633, 60, 80),
            'player_card2': (973, 623, 60, 80),
            'flop_cards': [(540, 450, 70, 90), (620, 450, 70, 90), (700, 450, 70, 90)],
            'turn_card': (780, 450, 70, 90),
            'river_card': (860, 450, 70, 90),
            'action_buttons': {
                'fold': (1230, 950, 120, 60),
                'check': (1350, 950, 120, 60),
                'bet': (1470, 950, 120, 60)
            }
        }

    def play_hand(self):
        """Stage-aware hand playing method that can handle any point in the hand"""
        print("Starting stage-aware hand analysis...")
        
        # Take screenshot
        img = self.screenshot_manager.capture_game_window()
        if img is None:
            print("Failed to take screenshot")
            return
        
        # Detect current game stage and available cards
        try:
            player_cards = self.card_detector.get_player_cards(img, self.screen_regions, self.game_window)
            community_cards = self.card_detector.get_community_cards(img, self.screen_regions, self.game_window)
            
            # Validate and clean cards
            player_cards = self._validate_and_clean_cards(player_cards)
            community_cards = self._validate_and_clean_cards(community_cards)
            
            # Determine current game stage
            game_stage = self._determine_game_stage(community_cards)
            print(f"Detected game stage: {game_stage}")
            print(f"Player cards: {player_cards}")
            print(f"Community cards: {community_cards}")
            
            # Validate that we have enough information
            if len(player_cards) < 2:
                print("Insufficient player cards detected")
                return
            
            # Stage-specific analysis and decision making
            if game_stage == "preflop":
                decision = self._handle_preflop(player_cards, community_cards)
            elif game_stage == "flop":
                decision = self._handle_flop(player_cards, community_cards)
            elif game_stage == "turn":
                decision = self._handle_turn(player_cards, community_cards)
            elif game_stage == "river":
                decision = self._handle_river(player_cards, community_cards)
            else:
                print(f"Unknown game stage: {game_stage}")
                decision = self._make_safe_decision(player_cards, community_cards)
            
            # Execute decision
            if decision:
                print(f"Stage-based decision: {decision}")
                self.logger.log_decision(decision)
                self._click_button(decision)
            else:
                print("No decision made, using fallback")
                self._make_safe_decision(player_cards, community_cards)
                
        except Exception as e:
            self.logger.log_error(f"Error in stage-aware play_hand: {e}", e)
            traceback.print_exc()
            print("Stage-aware analysis failed, using fallback")
            
            # Emergency fallback
            try:
                player_cards = self.simple_card_detection(img)
                if len(player_cards) >= 2:
                    emergency_decision = self._make_emergency_decision(player_cards)
                    self._click_button(emergency_decision)
                else:
                    self._click_button('check')
            except:
                self._click_button('check')

    def _determine_game_stage(self, community_cards):
        """Determine the current stage of the game based on community cards"""
        if not community_cards:
            return "preflop"
        elif len(community_cards) == 3:
            return "flop"
        elif len(community_cards) == 4:
            return "turn"
        elif len(community_cards) == 5:
            return "river"
        else:
            # Handle cases where we might have partial detection
            if community_cards:
                # Check which community card positions are filled
                flop_count = sum(1 for card in community_cards[:3] if card)
                if flop_count == 3:
                    if len(community_cards) >= 4 and community_cards[3]:
                        if len(community_cards) >= 5 and community_cards[4]:
                            return "river"
                        else:
                            return "turn"
                    else:
                        return "flop"
            return "preflop"

    def _handle_preflop(self, player_cards, community_cards):
        """Handle preflop decision making"""
        print("=== PREFLOP ANALYSIS ===")
        
        # Preflop is all about hole cards and position
        hand_strength = self._evaluate_preflop_hand(player_cards)
        
        # Calculate preflop odds (simplified - mainly based on hand strength)
        if hand_strength >= 8:  # Premium hands (AA, KK, QQ, AKs)
            win_prob = 0.75
            action = 'bet'
        elif hand_strength >= 6:  # Strong hands (JJ, TT, AQ, AK)
            win_prob = 0.60
            action = 'bet' if random.random() > 0.3 else 'call'
        elif hand_strength >= 4:  # Medium hands (99-77, AJ, KQ)
            win_prob = 0.45
            action = 'call' if random.random() > 0.5 else 'check'
        elif hand_strength >= 2:  # Weak hands (low pairs, weak aces)
            win_prob = 0.35
            action = 'check'
        else:  # Very weak hands
            win_prob = 0.25
            action = 'fold'
        
        print(f"Preflop hand strength: {hand_strength}/10")
        print(f"Estimated win probability: {win_prob:.2%}")
        print(f"Recommended action: {action}")
        
        return action

    def _handle_flop(self, player_cards, community_cards):
        """Handle flop decision making"""
        print("=== FLOP ANALYSIS ===")
        
        # Evaluate hand with flop cards
        hand_strength = self._evaluate_postflop_hand(player_cards, community_cards)
        
        # Calculate odds using simple calculator
        win_prob, lose_prob, tie_prob = self.odds_calculator.calculate_simple_odds(
            player_cards, community_cards
        )
        
        print(f"Flop hand strength: {hand_strength}/10")
        print(f"Calculated odds - Win: {win_prob:.2%}, Lose: {lose_prob:.2%}, Tie: {tie_prob:.2%}")
        
        # Flop-specific decision making
        if win_prob > 0.70:
            action = 'bet'  # Very strong hand, bet for value
        elif win_prob > 0.50:
            action = 'bet' if random.random() > 0.4 else 'call'  # Strong hand, sometimes bet
        elif win_prob > 0.35:
            action = 'call' if random.random() > 0.6 else 'check'  # Drawing hand
        else:
            action = 'fold' if random.random() > 0.3 else 'check'  # Weak hand
        
        print(f"Flop decision: {action}")
        return action

    def _handle_turn(self, player_cards, community_cards):
        """Handle turn decision making"""
        print("=== TURN ANALYSIS ===")
        
        # Evaluate hand with turn card
        hand_strength = self._evaluate_postflop_hand(player_cards, community_cards)
        
        # Calculate odds
        win_prob, lose_prob, tie_prob = self.odds_calculator.calculate_simple_odds(
            player_cards, community_cards
        )
        
        print(f"Turn hand strength: {hand_strength}/10")
        print(f"Calculated odds - Win: {win_prob:.2%}, Lose: {lose_prob:.2%}, Tie: {tie_prob:.2%}")
        
        # Turn-specific decision making (more conservative than flop)
        if win_prob > 0.75:
            action = 'bet'  # Very strong hand
        elif win_prob > 0.55:
            action = 'call'  # Good hand, but more conservative
        elif win_prob > 0.40:
            action = 'check'  # Marginal hand
        else:
            action = 'fold'  # Weak hand
        
        print(f"Turn decision: {action}")
        return action

    def _handle_river(self, player_cards, community_cards):
        """Handle river decision making"""
        print("=== RIVER ANALYSIS ===")
        
        # Final hand evaluation
        hand_strength = self._evaluate_postflop_hand(player_cards, community_cards)
        final_hand = self._identify_final_hand(player_cards, community_cards)
        
        # Calculate final odds
        win_prob, lose_prob, tie_prob = self.odds_calculator.calculate_simple_odds(
            player_cards, community_cards
        )
        
        print(f"River hand strength: {hand_strength}/10")
        print(f"Final hand: {final_hand}")
        print(f"Final odds - Win: {win_prob:.2%}, Lose: {lose_prob:.2%}, Tie: {tie_prob:.2%}")
        
        # River-specific decision making (most conservative)
        if win_prob > 0.80:
            action = 'bet'  # Nut hand, bet for value
        elif win_prob > 0.60:
            action = 'bet' if random.random() > 0.5 else 'call'  # Strong hand
        elif win_prob > 0.45:
            action = 'call' if random.random() > 0.7 else 'check'  # Medium hand
        else:
            action = 'check'  # Weak hand, just check or fold
        
        print(f"River decision: {action}")
        return action

    def _evaluate_preflop_hand(self, player_cards):
        """Evaluate preflop hand strength (0-10 scale)"""
        if len(player_cards) < 2:
            return 0
        
        rank1, suit1 = player_cards[0][0], player_cards[0][1]
        rank2, suit2 = player_cards[1][0], player_cards[1][1]
        
        # Premium hands
        premium_pairs = ['AA', 'KK', 'QQ']
        premium_ak = ['AK']
        
        # Strong hands
        strong_pairs = ['JJ', 'TT']
        strong_ak = ['AQ', 'AJ']
        
        # Medium hands
        medium_pairs = ['99', '88', '77']
        medium_hands = ['KQ', 'KJ', 'QJ']
        
        # Check for pairs
        if rank1 == rank2:
            hand = f"{rank1}{rank2}"
            if hand in premium_pairs:
                return 10
            elif hand in strong_pairs:
                return 8
            elif hand in medium_pairs:
                return 6
            else:
                return 4  # Low pairs
        
        # Check for suited cards
        is_suited = suit1 == suit2
        
        # Check for high card hands
        high_cards = ['A', 'K', 'Q', 'J']
        if rank1 in high_cards and rank2 in high_cards:
            # Sort ranks for consistent evaluation
            ranks = sorted([rank1, rank2], key=lambda x: self._get_rank_value(x), reverse=True)
            hand = f"{ranks[0]}{ranks[1]}"
            
            if hand in premium_ak:
                return 9 if is_suited else 8
            elif hand in strong_ak:
                return 7 if is_suited else 6
            elif hand in medium_hands:
                return 5 if is_suited else 4
            else:
                return 3 if is_suited else 2
        
        # Ace with any card
        if 'A' in [rank1, rank2]:
            return 3 if is_suited else 2
        
        # Default weak hands
        return 1

    def _evaluate_postflop_hand(self, player_cards, community_cards):
        """Evaluate postflop hand strength (0-10 scale)"""
        try:
            # Use the simple odds calculator's hand evaluation
            return self.odds_calculator._evaluate_hand_strength(player_cards, community_cards)
        except:
            # Fallback to basic evaluation
            return self._basic_hand_evaluation(player_cards, community_cards)

    def _basic_hand_evaluation(self, player_cards, community_cards):
        """Basic hand evaluation as fallback"""
        all_cards = player_cards + community_cards
        
        # Count ranks and suits
        rank_counts = {}
        suit_counts = {}
        
        for card in all_cards:
            if len(card) >= 2:
                rank = card[0]
                suit = card[1]
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        # Check for pairs, sets, etc.
        pairs = sum(1 for count in rank_counts.values() if count == 2)
        three_of_kind = sum(1 for count in rank_counts.values() if count == 3)
        four_of_kind = sum(1 for count in rank_counts.values() if count == 4)
        
        # Check for flush
        is_flush = max(suit_counts.values()) >= 5
        
        # Basic strength assessment
        if four_of_kind:
            return 9
        elif three_of_kind and pairs >= 1:
            return 8
        elif is_flush:
            return 7
        elif three_of_kind:
            return 6
        elif pairs >= 2:
            return 5
        elif pairs == 1:
            return 3
        else:
            # High card
            high_ranks = ['A', 'K', 'Q', 'J']
            high_cards = sum(1 for card in player_cards if card[0] in high_ranks)
            return min(high_cards, 2)

    def _identify_final_hand(self, player_cards, community_cards):
        """Identify the final hand type (for river stage)"""
        all_cards = player_cards + community_cards
        
        # Count ranks and suits
        rank_counts = {}
        suit_counts = {}
        
        for card in all_cards:
            if len(card) >= 2:
                rank = card[0]
                suit = card[1]
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        # Check for different hand types
        if 4 in rank_counts.values():
            return "Four of a Kind"
        elif 3 in rank_counts.values() and 2 in rank_counts.values():
            return "Full House"
        elif max(suit_counts.values()) >= 5:
            return "Flush"
        elif 3 in rank_counts.values():
            return "Three of a Kind"
        elif list(rank_counts.values()).count(2) >= 2:
            return "Two Pair"
        elif 2 in rank_counts.values():
            return "One Pair"
        else:
            return "High Card"

    def _get_rank_value(self, rank):
        """Get numerical value of rank"""
        rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        return rank_map.get(rank, int(rank) if rank.isdigit() else 14)

    def _make_safe_decision(self, player_cards, community_cards):
        """Make a safe decision based on current situation"""
        game_stage = self._determine_game_stage(community_cards)
        
        # Very conservative decisions
        if game_stage == "preflop":
            # Only play very strong hands preflop
            hand_strength = self._evaluate_preflop_hand(player_cards)
            if hand_strength >= 6:
                return 'call'
            else:
                return 'fold'
        else:
            # Postflop - be more conservative
            win_prob, _, _ = self.odds_calculator.calculate_simple_odds(player_cards, community_cards)
            if win_prob > 0.50:
                return 'call'
            else:
                return 'check'

    def _make_emergency_decision(self, player_cards):
        """Emergency decision when everything else fails"""
        if len(player_cards) >= 2:
            # Check for any pair or high cards
            if player_cards[0][0] == player_cards[1][0]:  # Pair
                return 'call'
            elif 'A' in [player_cards[0][0], player_cards[1][0]]:  # Ace
                return 'call'
        
        return 'check'
        
    def _validate_and_clean_cards(self, cards):
        """Validate and clean card list"""
        if not cards:
            return []
        
        cleaned_cards = []
        seen_cards = set()
        
        for card in cards:
            # Check if card is valid
            if card and self.card_detector._is_valid_card(card):
                # Check for duplicates
                if card not in seen_cards:
                    cleaned_cards.append(card)
                    seen_cards.add(card)
                else:
                    print(f"Duplicate card detected and removed: {card}")
            else:
                print(f"Invalid card detected and removed: {card}")
        
        return cleaned_cards

    def make_enhanced_decision(self, win_prob: float, lose_prob: float, tie_prob: float, pot_odds: float = 0.2) -> str:
        """Make enhanced decision based on comprehensive probability analysis"""
        
        # Calculate expected value
        expected_value = win_prob * 1 - lose_prob * 1 + tie_prob * 0
        
        # Adjust for pot odds
        adjusted_ev = expected_value - pot_odds
        
        # Enhanced decision logic
        if win_prob < 0.25:
            return 'fold'  # Very weak hand
        elif win_prob < 0.35:
            return 'check' if adjusted_ev < 0 else 'call'  # Weak hand
        elif win_prob < 0.45:
            return 'check'  # Marginal hand
        elif win_prob < 0.55:
            return 'call' if adjusted_ev > 0 else 'check'  # Medium hand
        elif win_prob < 0.70:
            return 'bet' if adjusted_ev > 0.1 else 'call'  # Strong hand
        else:
            return 'bet'  # Very strong hand

    def simple_card_detection(self, img):
        """Simple fallback card detection using basic template matching"""
        try:
            # Get player card regions
            x1, y1, w1, h1 = self.screen_regions['player_card1']
            x2, y2, w2, h2 = self.screen_regions['player_card2']
            
            # Convert to relative coordinates
            rel_x1 = x1 - self.game_window['left']
            rel_y1 = y1 - self.game_window['top']
            rel_x2 = x2 - self.game_window['left']
            rel_y2 = y2 - self.game_window['top']
            
            # Extract card images
            card1_img = img[rel_y1:rel_y1+h1, rel_x1:rel_x1+w1]
            card2_img = img[rel_y2:rel_y2+h2, rel_x2:rel_x2+w2]
            
            # Simple color-based detection
            cards = []
            
            for i, card_img in enumerate([card1_img, card2_img]):
                try:
                    # Convert to HSV
                    hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
                    
                    # Detect red/black
                    red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                    red_pixels = np.sum(red_mask > 0)
                    
                    is_red = red_pixels > 500
                    
                    # Simple rank detection based on image brightness patterns
                    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)
                    
                    # Very basic rank estimation (this is a fallback!)
                    if brightness > 180:
                        rank = 'A'  # Ace cards are usually brighter
                    elif brightness > 160:
                        rank = 'K'
                    elif brightness > 140:
                        rank = 'Q'
                    elif brightness > 120:
                        rank = 'J'
                    else:
                        rank = '2'  # Default to low card
                    
                    suit = 'h' if is_red else 's'
                    cards.append(f"{rank}{suit}")
                    
                except Exception as e:
                    print(f"Simple detection failed for card {i}: {e}")
                    cards.append(None)
            
            return cards
            
        except Exception as e:
            print(f"Simple card detection error: {e}")
            return []

    def _is_strong_hand(self, cards):
        """Check if hand is strong (pairs, high cards, etc.)"""
        if not cards or len(cards) < 2:
            return False
        
        # Check for pairs
        if cards[0][0] == cards[1][0]:
            return True
        
        # Check for high cards
        high_ranks = ['A', 'K', 'Q', 'J']
        if cards[0][0] in high_ranks and cards[1][0] in high_ranks:
            return True
        
        # Check for Ace with high card
        if 'A' in [cards[0][0], cards[1][0]]:
            other_card = cards[1] if cards[0][0] == 'A' else cards[0]
            if other_card[0] in ['K', 'Q', 'J']:
                return True
        
        return False

    def _is_weak_hand(self, cards):
        """Check if hand is weak"""
        if not cards or len(cards) < 2:
            return True
        
        # Check for low cards
        low_ranks = ['2', '3', '4', '5', '6', '7']
        weak_count = sum(1 for card in cards if card[0] in low_ranks)
        
        return weak_count >= 2

    def _click_button(self, action):
        """Click the specified action button"""
        if action in self.screen_regions['action_buttons']:
            x, y, w, h = self.screen_regions['action_buttons'][action]
            center_x = x + w // 2
            center_y = y + h // 2
            
            print(f"Clicking {action} button at ({center_x}, {center_y})")
            pyautogui.click(center_x, center_y)
            
            # Verify the click
            time.sleep(0.5)
            return True
        else:
            print(f"Unknown action: {action}")
            return False

    def test_regions(self):
        """Test screen regions"""
        print("Testing screen regions...")
        img = self.screenshot_manager.capture_game_window()
        if img is None:
            print("Failed to capture screen")
            return
        
        # Test player card regions
        for region_name, (x, y, w, h) in [
            ('player_card1', self.screen_regions['player_card1']),
            ('player_card2', self.screen_regions['player_card2'])
        ]:
            rel_x = x - self.game_window['left']
            rel_y = y - self.game_window['top']
            card_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
            cv2.imwrite(f'test_{region_name}.png', card_img)
            print(f"Saved {region_name} region to test_{region_name}.png")

    def test_card_detection(self):
        """Test card detection"""
        print("Testing card detection...")
        img = self.screenshot_manager.capture_game_window()
        if img is None:
            print("Failed to capture screen")
            return
        
        player_cards = self.card_detector.get_player_cards(img, self.screen_regions, self.game_window)
        community_cards = self.card_detector.get_community_cards(img, self.screen_regions, self.game_window)
        
        print(f"Detected player cards: {player_cards}")
        print(f"Detected community cards: {community_cards}")

    def calibrate_screen(self):
        """Calibrate screen regions"""
        print("Screen calibration not implemented yet")

    def preview_game_window(self):
        """Preview game window"""
        print("Previewing game window...")
        img = self.screenshot_manager.capture_game_window()
        if img is None:
            print("Failed to capture screen")
            return
        
        cv2.imshow('Game Window', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_bot(self, hands_to_play=10):
        """Run the bot continuously"""
        print(f"Running bot for {hands_to_play} hands...")
        
        for i in range(hands_to_play):
            print(f"\n=== Hand {i+1}/{hands_to_play} ===")
            self.play_hand()
            
            # Wait between hands
            time.sleep(2)


class SimpleOddsCalculator:
    """Simple odds calculator for fallback use"""
    
    def __init__(self):
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        self.rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        
    def calculate_simple_odds(self, player_cards, community_cards):
        """Calculate simple odds based on hand strength"""
        try:
            # Basic hand strength evaluation
            hand_strength = self._evaluate_hand_strength(player_cards, community_cards)
            
            # Convert strength to probabilities
            if hand_strength >= 8:  # Very strong hand
                return 0.85, 0.12, 0.03
            elif hand_strength >= 6:  # Strong hand
                return 0.70, 0.25, 0.05
            elif hand_strength >= 4:  # Medium hand
                return 0.50, 0.45, 0.05
            elif hand_strength >= 2:  # Weak hand
                return 0.30, 0.65, 0.05
            else:  # Very weak hand
                return 0.15, 0.80, 0.05
                
        except Exception as e:
            print(f"Simple odds calculation error: {e}")
            return 0.33, 0.64, 0.03  # Default probabilities
    
    def _evaluate_hand_strength(self, player_cards, community_cards):
        """Evaluate hand strength on a scale of 0-10"""
        if not player_cards or len(player_cards) < 2:
            return 0
        
        all_cards = player_cards + community_cards
        
        # Get rank values
        ranks = []
        suits = []
        for card in all_cards:
            if len(card) >= 2:
                rank_char = card[0]
                suit_char = card[1]
                
                # Get rank value
                rank_value = self.rank_values.get(rank_char, int(rank_char) if rank_char.isdigit() else 14)
                ranks.append(rank_value)
                suits.append(suit_char)
        
        # Check for pairs, three of a kind, etc.
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Check for flush
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        is_flush = max_suit_count >= 5
        
        # Check for straight
        sorted_ranks = sorted(set(ranks))
        is_straight = False
        straight_high = 0
        
        # Check for regular straight
        for i in range(len(sorted_ranks) - 4):
            if sorted_ranks[i+4] - sorted_ranks[i] == 4:
                is_straight = True
                straight_high = sorted_ranks[i+4]
                break
        
        # Check for A-2-3-4-5 straight
        if set([14, 2, 3, 4, 5]).issubset(set(ranks)):
            is_straight = True
            straight_high = 5
        
        # Evaluate hand strength
        if is_straight and is_flush:
            return 9  # Straight flush
        elif 4 in rank_counts.values():
            return 8  # Four of a kind
        elif 3 in rank_counts.values() and 2 in rank_counts.values():
            return 7  # Full house
        elif is_flush:
            return 6  # Flush
        elif is_straight:
            return 5  # Straight
        elif 3 in rank_counts.values():
            return 4  # Three of a kind
        elif list(rank_counts.values()).count(2) >= 2:
            return 3  # Two pair
        elif 2 in rank_counts.values():
            return 2  # One pair
        else:
            # High card
            high_card = max(ranks[:2])  # Only consider player cards for high card
            if high_card >= 14:  # Ace
                return 1
            elif high_card >= 13:  # King
                return 1
            elif high_card >= 12:  # Queen
                return 1
            else:
                return 0

    def preview_game_window(self):
        """Preview game window and capture on spacebar press"""
        print("=== Game Window Preview ===")
        print("Press SPACE to capture current frame")
        print("Press ESC to exit preview mode")
        self.logger.log("Starting game window preview")
        
        # Initialize window capture
        window_capture = GameWindowCapture("GOP3")
        
        # Try to find the game window
        if not window_capture.find_window():
            self.logger.log("Could not find game window automatically", level="WARNING")
            print("Trying all available windows...")
            
            if not window_capture.try_all_windows():
                self.logger.log("Could not find any suitable game window", level="ERROR")
                print("Please make sure the game is running and try again.")
                return
        
        # Activate the window
        if not window_capture.activate_window():
            self.logger.log("Could not activate game window", level="ERROR")
            print("Could not activate game window.")
            return
        
        # Create window for preview
        cv2.namedWindow("Game Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Game Preview", 800, 600)
        
        # Counter for captured images
        counter = 0
        
        # Status variables
        last_capture_time = 0
        capture_interval = 0.5  # Minimum time between captures
        
        self.logger.log("Starting preview loop")
        
        while True:
            try:
                # Check if window still exists
                if window_capture.window and not window_capture.window.isActive:
                    self.logger.log("Game window is no longer active", level="WARNING")
                    break
                
                # Capture game image
                current_time = time.time()
                if current_time - last_capture_time >= capture_interval:
                    game_image = window_capture.capture_game_image()
                    last_capture_time = current_time
                else:
                    game_image = None
                
                if game_image is not None:
                    # Show the image
                    cv2.imshow("Game Preview", game_image)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == 32:  # SPACE key
                        # Save the captured image
                        filename = f"game_capture_{counter}.png"
                        self.logger.save_image(game_image, filename, f"Game window capture #{counter}")
                        print(f"Captured image saved as {filename}")
                        counter += 1
                        
                    elif key == 27:  # ESC key
                        self.logger.log("Exiting preview mode")
                        print("Exiting preview mode")
                        break
                        
                    elif key == ord('r'):  # R key to refresh window
                        self.logger.log("Refreshing window")
                        if not window_capture.find_window():
                            self.logger.log("Could not find window after refresh", level="ERROR")
                            break
                        window_capture.activate_window()
                        
                else:
                    self.logger.log("Failed to capture game image", level="WARNING")
                    time.sleep(0.1)  # Wait before trying again
                    
            except KeyboardInterrupt:
                self.logger.log("Preview interrupted by user")
                print("Preview interrupted by user")
                break
                
            except Exception as e:
                self.logger.log_error(f"Error in preview loop: {e}", e)
                print(f"Error: {e}")
                time.sleep(1)  # Wait before trying again
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
        
        # Clean up
        cv2.destroyAllWindows()
        self.logger.log("Preview mode ended")

    def _setup_from_calibration(self):
        """Setup bot from loaded calibration data"""
        self.game_window = self.calibration_data['game_window']
        self.screen_regions = self.calibration_data['screen_regions']
        self.calibrated_coords = self.calibration_data['calibrated_coords']
        self.system_bars = self.calibration_data.get('system_bars', {
            'top_bar_height': 40,
            'bottom_bar_height': 40,
            'side_margin': 0
        })
        
        self.logger.log("Loaded calibration data:")
        self.logger.log(f"  Game window: {self.game_window['width']}x{self.game_window['height']} at ({self.game_window['left']}, {self.game_window['top']})")
        self.logger.log(f"  System bars: {self.system_bars}")
        self.logger.log(f"  Screen regions: {len(self.screen_regions)} regions defined")

    def _setup_defaults(self):
        """Setup default values if no calibration file"""
        self.system_bars = {
            'top_bar_height': 40,
            'bottom_bar_height': 40,
            'side_margin': 0
        }
        self.game_window = {
            'left': 0,
            'top': self.system_bars['top_bar_height'],
            'width': 1920,
            'height': 1080 - self.system_bars['top_bar_height'] - self.system_bars['bottom_bar_height']
        }
        self.screen_regions = self._get_default_regions()
        self.calibrated_coords = {}
        
        self.logger.log("Using default calibration values")
        self.logger.log(f"  Game window: {self.game_window['width']}x{self.game_window['height']} at ({self.game_window['left']}, {self.game_window['top']})")   

    def _get_default_regions(self):
        """Get default screen regions"""
        return {
            'game_window': (2, 24, 1913, 1008),
            'player_card1': (929, 673, 60, 80),
            'player_card2': (1003, 663, 60, 80),
            'flop_cards': [
                (778, 441, 70, 90),
                (870, 441, 70, 90),
                (964, 441, 70, 90)
            ],
            'turn_card': (1061, 441, 70, 90),
            'river_card': (1139, 441, 70, 90),
            'action_buttons': {
                'fold': (736, 976, 120, 60),
                'check': (1089, 976, 120, 60),
                'bet': (1472, 976, 120, 60),
            },
            'chips_display': (50, 50, 200, 50),
            'timer': (1700, 50, 150, 50),
        }
    
    def test_regions(self):
        """Test screen regions by drawing rectangles on screenshot"""
        self.logger.log("Testing screen regions")
        
        # Take full screen screenshot
        full_screen_img = self.screenshot_manager.capture_full_screen()
        if full_screen_img is None:
            self.logger.log("Failed to take full screen screenshot", level="ERROR")
            return
        
        # Draw regions
        test_img = self._draw_regions_on_image(full_screen_img)
        
        # Save images
        self.logger.save_image(test_img, "region_test_full.png", "Full screen region test")
        self.logger.log("Full screen region test image saved as 'region_test_full.png'")
        
        # Save game window only
        gw_left = self.game_window['left']
        gw_top = self.game_window['top']
        gw_right = gw_left + self.game_window['width']
        gw_bottom = gw_top + self.game_window['height']
        
        game_img = test_img[gw_top:gw_bottom, gw_left:gw_right]
        self.logger.save_image(game_img, "region_test_game.png", "Game window only region test")
        self.logger.log("Game window only test image saved as 'region_test_game.png'")
        
        # Show images
        try:
            cv2.imshow('Screen Regions Test (Full Screen)', test_img)
            cv2.imshow('Screen Regions Test (Game Window)', game_img)
            print("Press any key to close image windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            self.logger.log("Could not display images. Check saved files.", level="WARNING")

    def test_card_detection(self):
        """Test card detection without playing"""
        self.logger.log("Testing card detection")
        
        # Take screenshot
        img = self.screenshot_manager.capture_game_window()
        if img is None:
            self.logger.log("Failed to take screenshot", level="ERROR")
            return
        
        # Detect cards
        player_cards = self.card_detector.get_player_cards(img, self.screen_regions, self.game_window)
        community_cards = self.card_detector.get_community_cards(img, self.screen_regions, self.game_window)
        
        # Log card detection
        self.logger.log_card_detection(player_cards, community_cards)
        
        # Validate cards
        player_cards, community_cards = self.card_detector.validate_cards(player_cards, community_cards)
        
        print(f"Detected player cards: {player_cards}")
        print(f"Detected community cards: {community_cards}")
        
        # Test card conversion
        for card in player_cards + community_cards:
            try:
                rank, suit = self.game_simulator.card_to_values(card)
                print(f"Card {card} -> Rank: {rank}, Suit: {suit}")
                self.logger.log(f"Card conversion test: {card} -> Rank: {rank}, Suit: {suit}")
            except Exception as e:
                self.logger.log_error(f"Error converting card {card}", e)

    def calibrate_screen(self):
        """Interactive calibration to find exact screen positions"""
        self.logger.log("Starting screen calibration")
        print("=== Screen Calibration Mode ===")
        print("Move your mouse to the specified locations and press ENTER when ready")
        
        # Import here to avoid circular imports
        from utils import ScreenCalibrator
        
        calibrator = ScreenCalibrator()
        calibrated_coords, calibrated_window = calibrator.calibrate()
        
        # Update regions based on calibration
        self._update_regions_from_calibration(calibrated_coords)
        
        # Save calibration
        self._save_calibration(calibrated_coords, calibrated_window)
        
        print("Calibration complete!")
        self.logger.log("Screen calibration completed successfully")

    def _draw_regions_on_image(self, img):
        """Draw all regions on image"""
        test_img = img.copy()
        
        # Draw game window boundary
        gw_left = self.game_window['left']
        gw_top = self.game_window['top']
        gw_right = gw_left + self.game_window['width']
        gw_bottom = gw_top + self.game_window['height']
        
        cv2.rectangle(test_img, (gw_left, gw_top), (gw_right, gw_bottom), (255, 255, 255), 4)
        cv2.putText(test_img, 'GAME WINDOW', (gw_left, gw_top - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw calibrated points
        if self.calibrated_coords:
            for point_name, coords in self.calibrated_coords.items():
                x, y = coords
                cv2.circle(test_img, (x, y), 8, (0, 255, 255), -1)
                cv2.putText(test_img, point_name, (x + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw card regions
        colors = {
            'player_card1': (255, 0, 0),
            'player_card2': (0, 255, 0),
            'flop_cards': [(0, 0, 255), (255, 255, 0), (255, 0, 255)],
            'turn_card': (0, 255, 255),
            'river_card': (128, 128, 128),
        }
        
        # Draw player cards
        for card_name, color in [('player_card1', colors['player_card1']), 
                                ('player_card2', colors['player_card2'])]:
            x, y, w, h = self.screen_regions[card_name]
            x2, y2 = x + w, y + h
            cv2.rectangle(test_img, (x, y), (x2, y2), color, 2)
            cv2.putText(test_img, card_name, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw flop cards
        for i, (coords, color) in enumerate(zip(self.screen_regions['flop_cards'], colors['flop_cards'])):
            x, y, w, h = coords
            x2, y2 = x + w, y + h
            cv2.rectangle(test_img, (x, y), (x2, y2), color, 2)
            cv2.putText(test_img, f'flop_{i+1}', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw turn and river
        for card_name, color in [('turn_card', colors['turn_card']), 
                                ('river_card', colors['river_card'])]:
            x, y, w, h = self.screen_regions[card_name]
            x2, y2 = x + w, y + h
            cv2.rectangle(test_img, (x, y), (x2, y2), color, 2)
            cv2.putText(test_img, card_name, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw action buttons
        button_colors = {
            'fold': (0, 0, 255),
            'check': (0, 255, 0),
            'bet': (255, 255, 0),
        }
        
        for button_name, color in button_colors.items():
            x, y, w, h = self.screen_regions['action_buttons'][button_name]
            x2, y2 = x + w, y + h
            cv2.rectangle(test_img, (x, y), (x2, y2), color, 3)
            cv2.putText(test_img, button_name.upper(), (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return test_img

    def _update_regions_from_calibration(self, calibrated_coords):
        """Update screen regions based on calibration points"""
        self.logger.log("Updating screen regions from calibration")
        
        # Import here to avoid circular imports
        from utils import RegionUpdater
        
        updater = RegionUpdater()
        self.screen_regions = updater.update_regions(calibrated_coords, self.screen_regions)
        
        self.logger.log("Screen regions updated successfully")

    def _save_calibration(self, calibrated_coords, calibrated_window):
        """Save calibration data to file"""
        calibration_data = {
            'screen_resolution': (1920, 1080),
            'system_bars': self.system_bars,
            'game_window': self.game_window,
            'calibrated_coords': calibrated_coords,
            'calibrated_window': calibrated_window,
            'screen_regions': self.screen_regions
        }
        
        with open(self.calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        self.logger.save_calibration_data(calibration_data)
        self.logger.log(f"Calibration saved to {self.calibration_file}")