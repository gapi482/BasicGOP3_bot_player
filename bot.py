#!/usr/bin/env python3
"""
Main Bot Class for Governor of Poker - Updated with Improved Card Detection
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
import config
from game_simulation import GameSimulator
from utils import GameWindowCapture, WindowDetector, ScreenshotManager
from card_confirmation import confirm_cards
import config

class GovernorOfPokerBot:
    def __init__(self, calibration_file='screen_calibration.json', logger=None):
        """Initialize the bot with calibration data"""
        self.calibration_file = calibration_file
        self.calibration_data = None
        
        # Initialize logger
        self.logger = Logger()
        self.logger.log("Initializing Governor of Poker Bot with Improved Card Detection")
        
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
        self.card_detector = CardDetector()  # Now using improved template matching
        self.game_simulator = GameSimulator()
        
        # Initialize simple odds calculator (fallback)
        self.odds_calculator = SimpleOddsCalculator()
        
        # Safety settings
        pyautogui.PAUSE = 0.5
        pyautogui.FAILSAFE = True
        
        self.logger.log("Bot initialized successfully with improved card detection")
    
    def _setup_from_calibration(self):
        """Setup from calibration data"""
        self.game_window = self.calibration_data['game_window']
        self.screen_regions = self.calibration_data['screen_regions']
    
    def _setup_defaults(self):
        """Setup default values"""
        self.game_window = dict(config.DEFAULT_GAME_WINDOW)
        self.screen_regions = dict(config.DEFAULT_SCREEN_REGIONS)
    
    def test_card_detection(self):
        """Test the improved card detection system on actual game screen"""
        # Activate the game window first
        self.window_detector.activate_game_window()
        
        # Take a screenshot of the game window
        screenshot = self.screenshot_manager.capture_game_window()
        if screenshot is None:
            self.logger.log("Failed to capture screen for card detection test", level="ERROR")
            return
        
        # Save the screenshot for debugging
        cv2.imwrite('game_screenshot_test.png', screenshot)
        
        # Test player cards
        player_cards = self.card_detector.get_player_cards(screenshot, self.screen_regions, self.game_window)
        
        # Test community cards
        community_cards = self.card_detector.get_community_cards(screenshot, self.screen_regions, self.game_window)
        
        # Display results
        print(f"\n=== Card Detection Test Results ===")
        print(f"Player Cards: {player_cards}")
        print(f"Community Cards: {community_cards}")
        print(f"Total Cards Detected: {len(player_cards) + len(community_cards)}")
        
        # Detailed analysis of each card region
        print(f"\n=== Detailed Card Analysis ===")
        
        # Analyze player cards
        for i, region_name in enumerate(['player_card1', 'player_card2']):
            x, y, w, h = self.screen_regions[region_name]
            rel_x = x - self.game_window['left']
            rel_y = y - self.game_window['top']
            
            # Extract card image
            card_img = screenshot[rel_y:rel_y+h, rel_x:rel_x+w]
            
            # Save extracted card image for debugging
            cv2.imwrite(f'extracted_{region_name}.png', card_img)
            
            # Check if card is present
            is_present = self.card_detector._is_card_present(card_img)
            print(f"\n{region_name}:")
            print(f"  Position: ({rel_x}, {rel_y}) Size: {w}x{h}")
            print(f"  Card present: {is_present}")
            
            if is_present:
                # Test card detection on this specific image
                matched_card = self.card_detector._detect_card(card_img, region_name)
                print(f"  Detected: {matched_card}")
                
                # Debug: Show top template matches
                debug_results = self.card_detector.debug_card_detection(card_img, region_name)
                if 'top_matches' in debug_results:
                    print(f"  Top 5 template matches:")
                    for j, (template_name, score) in enumerate(debug_results['top_matches'][:5]):
                        print(f"    {j+1}. {template_name}: {score:.3f}")
            else:
                print(f"  Detection skipped - no card present")
            
            print(f"  Image saved as: extracted_{region_name}.png")
        
        # Analyze community cards
        for i, coords in enumerate(self.screen_regions['flop_cards']):
            x, y, w, h = coords
            rel_x = x - self.game_window['left']
            rel_y = y - self.game_window['top']
            region_name = f"flop_{i+1}"
            
            # Extract card image
            card_img = screenshot[rel_y:rel_y+h, rel_x:rel_x+w]
            
            # Save extracted card image for debugging
            cv2.imwrite(f'extracted_{region_name}.png', card_img)
            
            # Check if card is present
            is_present = self.card_detector._is_card_present(card_img)
            print(f"\n{region_name}:")
            print(f"  Position: ({rel_x}, {rel_y}) Size: {w}x{h}")
            print(f"  Card present: {is_present}")
            
            if is_present:
                # Test card detection on this specific image
                matched_card = self.card_detector._detect_card(card_img, region_name)
                print(f"  Detected: {matched_card}")
                
                # Debug: Show top template matches
                debug_results = self.card_detector.debug_card_detection(card_img, region_name)
                if 'top_matches' in debug_results:
                    print(f"  Top 5 template matches:")
                    for j, (template_name, score) in enumerate(debug_results['top_matches'][:5]):
                        print(f"    {j+1}. {template_name}: {score:.3f}")
            else:
                print(f"  Detection skipped - no card present")
            
            print(f"  Image saved as: extracted_{region_name}.png")
        
        # Analyze turn card
        x, y, w, h = self.screen_regions['turn_card']
        rel_x = x - self.game_window['left']
        rel_y = y - self.game_window['top']
        region_name = "turn"
        
        # Extract card image
        card_img = screenshot[rel_y:rel_y+h, rel_x:rel_x+w]
        
        # Save extracted card image for debugging
        cv2.imwrite(f'extracted_{region_name}.png', card_img)
        
        # Check if card is present
        is_present = self.card_detector._is_card_present(card_img)
        print(f"\n{region_name}:")
        print(f"  Position: ({rel_x}, {rel_y}) Size: {w}x{h}")
        print(f"  Card present: {is_present}")
        
        if is_present:
            # Test card detection on this specific image
            matched_card = self.card_detector._detect_card(card_img, region_name)
            print(f"  Detected: {matched_card}")
            
            # Debug: Show top template matches
            debug_results = self.card_detector.debug_card_detection(card_img, region_name)
            if 'top_matches' in debug_results:
                print(f"  Top 5 template matches:")
                for j, (template_name, score) in enumerate(debug_results['top_matches'][:5]):
                    print(f"    {j+1}. {template_name}: {score:.3f}")
        else:
            print(f"  Detection skipped - no card present")
        
        print(f"  Image saved as: extracted_{region_name}.png")
        
        # Analyze river card
        x, y, w, h = self.screen_regions['river_card']
        rel_x = x - self.game_window['left']
        rel_y = y - self.game_window['top']
        region_name = "river"
        
        # Extract card image
        card_img = screenshot[rel_y:rel_y+h, rel_x:rel_x+w]
        
        # Save extracted card image for debugging
        cv2.imwrite(f'extracted_{region_name}.png', card_img)
        
        # Check if card is present
        is_present = self.card_detector._is_card_present(card_img)
        print(f"\n{region_name}:")
        print(f"  Position: ({rel_x}, {rel_y}) Size: {w}x{h}")
        print(f"  Card present: {is_present}")
        
        if is_present:
            # Test card detection on this specific image
            matched_card = self.card_detector._detect_card(card_img, region_name)
            print(f"  Detected: {matched_card}")
            
            # Debug: Show top template matches
            debug_results = self.card_detector.debug_card_detection(card_img, region_name)
            if 'top_matches' in debug_results:
                print(f"  Top 5 template matches:")
                for j, (template_name, score) in enumerate(debug_results['top_matches'][:5]):
                    print(f"    {j+1}. {template_name}: {score:.3f}")
        else:
            print(f"  Detection skipped - no card present")
        
        print(f"  Image saved as: extracted_{region_name}.png")
        
        # Create a visual test image with regions marked
        test_img = screenshot.copy()
        
        # Draw player card regions
        for i, region_name in enumerate(['player_card1', 'player_card2']):
            x, y, w, h = self.screen_regions[region_name]
            rel_x = x - self.game_window['left']
            rel_y = y - self.game_window['top']
            
            # Extract card image to check if present
            card_img = screenshot[rel_y:rel_y+h, rel_x:rel_x+w]
            is_present = self.card_detector._is_card_present(card_img)
            
            # Use green if card is present, red if not
            color = (0, 255, 0) if is_present else (0, 0, 255)
            
            cv2.rectangle(test_img, (rel_x, rel_y), (rel_x + w, rel_y + h), color, 2)
            
            # Add label
            matched = player_cards[i] if i < len(player_cards) else "None"
            status = "Present" if is_present else "Absent"
            cv2.putText(test_img, f"{region_name}: {matched} ({status})", 
                    (rel_x, rel_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw community card regions
        for i, coords in enumerate(self.screen_regions['flop_cards']):
            x, y, w, h = coords
            rel_x = x - self.game_window['left']
            rel_y = y - self.game_window['top']
            region_name = f"flop_{i+1}"
            
            # Extract card image to check if present
            card_img = screenshot[rel_y:rel_y+h, rel_x:rel_x+w]
            is_present = self.card_detector._is_card_present(card_img)
            
            # Use blue if card is present, red if not
            color = (255, 0, 0) if is_present else (0, 0, 255)
            
            cv2.rectangle(test_img, (rel_x, rel_y), (rel_x + w, rel_y + h), color, 2)
            
            # Add label
            matched = community_cards[i] if i < len(community_cards) else "None"
            status = "Present" if is_present else "Absent"
            cv2.putText(test_img, f"{region_name}: {matched} ({status})", 
                    (rel_x, rel_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw turn and river
        for i, region_name in enumerate(['turn_card', 'river_card']):
            x, y, w, h = self.screen_regions[region_name]
            rel_x = x - self.game_window['left']
            rel_y = y - self.game_window['top']
            
            # Extract card image to check if present
            card_img = screenshot[rel_y:rel_y+h, rel_x:rel_x+w]
            is_present = self.card_detector._is_card_present(card_img)
            
            # Use blue if card is present, red if not
            color = (255, 0, 0) if is_present else (0, 0, 255)
            
            cv2.rectangle(test_img, (rel_x, rel_y), (rel_x + w, rel_y + h), color, 2)
            
            # Add label
            matched = community_cards[3+i] if (3+i) < len(community_cards) else "None"
            status = "Present" if is_present else "Absent"
            label = "turn" if region_name == 'turn_card' else "river"
            cv2.putText(test_img, f"{label}: {matched} ({status})", 
                    (rel_x, rel_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save the annotated image
        cv2.imwrite('card_detection_test_results.png', test_img)
        print(f"\n=== Visual Test Results ===")
        print(f"Annotated screenshot saved as: card_detection_test_results.png")
        print(f"Open this file to see the detected cards and their presence status")
        
        # Summary
        print(f"\n=== Summary ===")
        print(f"Total player cards detected: {len(player_cards)}/2")
        print(f"Total community cards detected: {len(community_cards)}/5")
        print(f"Total cards detected: {len(player_cards) + len(community_cards)}/7")
        print(f"\nGenerated files:")
        print(f"  - game_screenshot_test.png (full screenshot)")
        print(f"  - card_detection_test_results.png (with detection results)")
        print(f"  - extracted_*.png (individual card images)")
        print(f"\nCheck the extracted_*.png files to see what the bot is trying to match")
        print(f"Green/Blue rectangles indicate cards that are present")
        print(f"Red rectangles indicate areas where no card was detected")

    def play_hand(self):
        """Play a single hand of poker"""
        # Activate the game window first
        self.window_detector.activate_game_window()
        
        # Take screenshot
        screenshot = self.screenshot_manager.capture_game_window()
        if screenshot is None:
            self.logger.log("Failed to capture screen", level="ERROR")
            return
        
        # Detect cards using improved template matching
        player_cards = self.card_detector.get_player_cards(screenshot, self.screen_regions, self.game_window)
        community_cards = self.card_detector.get_community_cards(screenshot, self.screen_regions, self.game_window)
        
        # Show confirmation window if enabled
        if config.BOT_BEHAVIOR.get('enable_card_confirmation', True):
            confirmation_result = confirm_cards(player_cards, community_cards)
            
            if confirmation_result['action'] == 'fold':
                self._take_action('fold')
                return
            elif confirmation_result['action'] == 'skip':
                # Use detected cards as-is
                final_player_cards = player_cards
                final_community_cards = community_cards
            else:
                # Use confirmed/corrected cards
                final_player_cards = confirmation_result['player_cards']
                final_community_cards = confirmation_result['community_cards']
        else:
            # Use detected cards directly without confirmation
            final_player_cards = player_cards
            final_community_cards = community_cards
        
        # Simple decision making based on confirmed cards
        if len(final_player_cards) == 2:
            # Calculate hand strength
            all_cards = final_player_cards + final_community_cards
            hand_strength = self._evaluate_hand_strength(all_cards)
            
            # Make decision based on hand strength
            if hand_strength > 0.7:
                self._take_action('raise')
            elif hand_strength > 0.4:
                self._take_action('call')
            else:
                self._take_action('fold')
        else:
            self.logger.log("Can't decide - player hands not recognized properly", level="WARNING")
            self._take_action('fold')
    
    def _evaluate_hand_strength(self, cards):
        """Simple hand strength evaluation"""
        # This is a simplified version - you can integrate with your odds calculator
        if not cards:
            return 0.0
        
        # Basic hand strength calculation
        strength = 0.0
        
        # Check for pairs, high cards, etc.
        ranks = [card[0] for card in cards if card]
        
        # High cards bonus
        high_cards = ['A', 'K', 'Q', 'J']
        for rank in ranks:
            if rank in high_cards:
                strength += 0.2
        
        # Pair bonus
        if len(ranks) == len(set(ranks)):
            strength += 0.3
        
        return min(strength, 1.0)
    
    def _take_action(self, action):
        """Take a poker action"""
        buttons = self.screen_regions.get('action_buttons', {})
        target = buttons.get(action)
        
        # Fallback mappings if specific action not present
        if target is None:
            fallback_order = {
                'call': ['check', 'raise', 'all_in', 'fold'],
                'raise': ['bet', 'all_in', 'call', 'check'],
                'check': ['call', 'fold'],
                'fold': [],
                'all_in': ['raise', 'bet']
            }
            for alt in fallback_order.get(action, []):
                if alt in buttons:
                    self.logger.log(f"Fallback: mapping '{action}' to '{alt}'", level="WARNING")
                    target = buttons[alt]
                    break
        
        if target is not None:
            x, y, w, h = target
            # Add some randomness to avoid detection
            x_offset = random.randint(-5, 5)
            y_offset = random.randint(-5, 5)
            pyautogui.click(x + x_offset, y + y_offset)
            time.sleep(1)  # Wait for action to complete
        else:
            self.logger.log(f"No available button for action: {action}", level="ERROR")


    def test_regions(self):
        """Test screen regions"""
        # Activate the game window first
        self.window_detector.activate_game_window()
        
        screenshot = self.screenshot_manager.capture_game_window()
        
        if screenshot is not None:
            # Draw rectangles around regions
            test_img = screenshot.copy()
            
            # Draw player card regions
            for region_name in ['player_card1', 'player_card2']:
                x, y, w, h = self.screen_regions[region_name]
                rel_x = x - self.game_window['left']
                rel_y = y - self.game_window['top']
                cv2.rectangle(test_img, (rel_x, rel_y), (rel_x + w, rel_y + h), (0, 255, 0), 2)
            
            # Draw community card regions
            for coords in self.screen_regions['flop_cards']:
                x, y, w, h = coords
                rel_x = x - self.game_window['left']
                rel_y = y - self.game_window['top']
                cv2.rectangle(test_img, (rel_x, rel_y), (rel_x + w, rel_y + h), (255, 0, 0), 2)
            
            # Draw turn and river
            for region_name in ['turn_card', 'river_card']:
                x, y, w, h = self.screen_regions[region_name]
                rel_x = x - self.game_window['left']
                rel_y = y - self.game_window['top']
                cv2.rectangle(test_img, (rel_x, rel_y), (rel_x + w, rel_y + h), (0, 0, 255), 2)
            
            # Save test image
            cv2.imwrite('test_regions.png', test_img)
            print("Test regions image saved as test_regions.png")
    
    def calibrate_screen(self):
        """Calibrate screen regions"""
        print("Screen calibration not implemented in this version")
        print("Please manually edit the screen_calibration.json file")
    
    def preview_game_window(self):
        """Preview the game window"""
        # Activate the game window first
        self.window_detector.activate_game_window()
        
        screenshot = self.screenshot_manager.capture_game_window()
        
        if screenshot is not None:
            cv2.imshow('Game Window Preview', screenshot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            self.logger.log("Failed to capture game window", level="ERROR")


class SimpleOddsCalculator:
    """Simple odds calculator for fallback"""
    
    def calculate_win_probability(self, player_cards, community_cards):
        """Calculate simple win probability"""
        # This is a placeholder - implement proper odds calculation
        if not player_cards:
            return 0.0
        
        # Basic calculation based on card values
        strength = 0.0
        for card in player_cards:
            if card:
                rank = card[0]
                if rank == 'A':
                    strength += 0.3
                elif rank in ['K', 'Q', 'J']:
                    strength += 0.2
                elif rank.isdigit() and int(rank) >= 10:
                    strength += 0.1
        
        return min(strength, 1.0)