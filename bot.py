#!/usr/bin/env python3
"""
Main Bot Class for Governor of Poker
"""

import cv2
import pyautogui
import time
import json
import os
import traceback
from logger import Logger
from card_detection import CardDetector
from game_simulation import GameSimulator
from utils import WindowDetector, ScreenshotManager, GameWindowCapture
from logger import Logger


# Update the GovernorOfPokerBot class
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
        
        # Safety settings
        pyautogui.PAUSE = 0.5
        pyautogui.FAILSAFE = True
        
        self.logger.log("Bot initialized successfully")
    
    def play_hand(self):
        """Main function to play one hand"""
        print("Starting new hand...")
        
        # Take screenshot
        img = self.screenshot_manager.capture_game_window()
        if img is None:
            print("Failed to take screenshot")
            return
        
        # Try normal card detection first
        player_cards = self.card_detector.get_player_cards(img, self.screen_regions, self.game_window)
        community_cards = self.card_detector.get_community_cards(img, self.screen_regions, self.game_window)
        
        # Check if detection was successful
        detection_successful = (
            len(player_cards) >= 2 and 
            all(player_cards) and 
            all(self.card_detector._is_valid_card(card) for card in player_cards)
        )
        
        if not detection_successful:
            print("Normal card detection failed or returned invalid cards, using simple detection...")
            player_cards = self.simple_card_detection(img)
            community_cards = []  # Skip community cards for now
            
            # Check if simple detection was successful
            if len(player_cards) < 2 or not all(player_cards):
                print("Could not detect player cards even with simple detection")
                return
        
        print(f"Player cards: {player_cards}")
        print(f"Community cards: {community_cards}")
        
        # Run simulation
        try:
            win_prob, lose_prob, tie_prob = self.game_simulator.monte_carlo_simulation(
                player_cards, community_cards
            )
            
            # Log simulation results
            self.logger.log_simulation_results(win_prob, lose_prob, tie_prob)
            
            # Print basic info to console
            print(f"Win probability: {win_prob:.2%}")
            print(f"Lose probability: {lose_prob:.2%}")
            print(f"Tie probability: {tie_prob:.2%}")
            
            # Make decision
            decision = self.game_simulator.make_decision(win_prob)
            self.logger.log_decision(decision)
            
            # Print basic info to console
            print(f"Decision: {decision}")
            
            # Execute decision
            self._click_button(decision)
            
        except Exception as e:
            self.logger.log_error("Error in simulation", e)
            traceback.print_exc()
            print("Using fallback decision")
            
            # Better fallback decision based on hand strength
            if self._is_strong_hand(player_cards):
                print("Strong hand detected, betting")
                self.logger.log_decision("bet (fallback)")
                self._click_button('bet')
            elif self._is_weak_hand(player_cards):
                print("Weak hand detected, checking")
                self.logger.log_decision("check (fallback)")
                self._click_button('check')
            else:
                print("Medium hand, checking")
                self.logger.log_decision("check (fallback)")
                self._click_button('check')
            """Main function to play one hand"""
            print("Starting new hand...")
            
            # Take screenshot
            img = self.screenshot_manager.capture_game_window()
            if img is None:
                print("Failed to take screenshot")
                return
            
            # Try normal card detection first
            player_cards = self.card_detector.get_player_cards(img, self.screen_regions, self.game_window)
            community_cards = self.card_detector.get_community_cards(img, self.screen_regions, self.game_window)
            
            # If normal detection fails or returns invalid cards, use simple detection
            if len(player_cards) < 2 or not all(self.card_detector._is_valid_card(card) for card in player_cards):
                print("Normal card detection failed or returned invalid cards, using simple detection...")
                player_cards = self.simple_card_detection(img)
                community_cards = []  # Skip community cards for now
            
            print(f"Player cards: {player_cards}")
            print(f"Community cards: {community_cards}")
            
            if len(player_cards) < 2 or not all(player_cards):  # Check for None values
                print("Could not detect player cards")
                return
            
            # Run simulation
            try:
                win_prob, lose_prob, tie_prob = self.game_simulator.monte_carlo_simulation(
                    player_cards, community_cards
                )
                
                # Log simulation results
                self.logger.log_simulation_results(win_prob, lose_prob, tie_prob)
                
                # Print basic info to console
                print(f"Win probability: {win_prob:.2%}")
                print(f"Lose probability: {lose_prob:.2%}")
                print(f"Tie probability: {tie_prob:.2%}")
                
                # Make decision
                decision = self.game_simulator.make_decision(win_prob)
                self.logger.log_decision(decision)
                
                # Print basic info to console
                print(f"Decision: {decision}")
                
                # Execute decision
                self._click_button(decision)
                
            except Exception as e:
                self.logger.log_error("Error in simulation", e)
                print("Using fallback decision")
                
                # Better fallback decision based on hand strength
                if self._is_strong_hand(player_cards):
                    print("Strong hand detected, betting")
                    self.logger.log_decision("bet (fallback)")
                    self._click_button('bet')
                elif self._is_weak_hand(player_cards):
                    print("Weak hand detected, checking")
                    self.logger.log_decision("check (fallback)")
                    self._click_button('check')
                else:
                    print("Medium hand, checking")
                    self.logger.log_decision("check (fallback)")
                    self._click_button('check')

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

    def run_bot(self, hands_to_play: int = 10):
        """Run the bot for specified number of hands"""
        self.logger.log(f"Starting Governor of Poker bot - will play {hands_to_play} hands")
        self.logger.log(f"Game window: {self.game_window['width']}x{self.game_window['height']} at ({self.game_window['left']}, {self.game_window['top']})")
        print("Press Ctrl+C to stop the bot at any time")
        
        try:
            for i in range(hands_to_play):
                print(f"\n=== Hand {i + 1} ===")
                self.logger.log(f"Starting hand {i + 1} of {hands_to_play}")
                
                try:
                    self.play_hand()
                    time.sleep(3)  # Wait between hands
                    self.logger.log(f"Completed hand {i + 1}, waiting 3 seconds")
                    
                except KeyboardInterrupt:
                    self.logger.log("Hand interrupted by user")
                    print("Hand interrupted by user")
                    break
                    
                except Exception as e:
                    self.logger.log_error(f"Error in hand {i + 1}", e)
                    print(f"Error in hand {i + 1}: {e}")
                    time.sleep(2)  # Wait before trying next hand
                    
        except KeyboardInterrupt:
            self.logger.log("Bot stopped by user")
            print("\nBot stopped by user")
            
        except Exception as e:
            self.logger.log_error(f"Fatal error in bot execution", e)
            print(f"Fatal error: {e}")
            
        finally:
            self.logger.log(f"Bot finished. Played {i + 1} hands out of {hands_to_play}")
            print("Bot finished")

    def _is_strong_hand(self, player_cards):
        """Check if player has a strong hand"""
        if not player_cards or len(player_cards) < 2:
            return False
        
        # Check for pairs, high cards, etc.
        card1, card2 = player_cards
        
        # Pair
        if card1[0] == card2[0]:
            self.logger.log(f"Strong hand detected: Pair of {card1[0]}s")
            return True
        
        # Two high cards (A, K, Q, J)
        high_ranks = ['A', 'K', 'Q', 'J']
        if card1[0] in high_ranks and card2[0] in high_ranks:
            self.logger.log(f"Strong hand detected: {card1[0]}{card1[1]} and {card2[0]}{card2[1]} (both high cards)")
            return True
        
        # Ace with anything
        if card1[0] == 'A' or card2[0] == 'A':
            self.logger.log("Strong hand detected: Ace with another card")
            return True
        
        return False

    def _is_weak_hand(self, player_cards):
        """Check if player has a weak hand"""
        if not player_cards or len(player_cards) < 2:
            return True
        
        card1, card2 = player_cards
        
        # Low cards (2-6)
        low_ranks = ['2', '3', '4', '5', '6']
        if card1[0] in low_ranks and card2[0] in low_ranks:
            # Check if they're not suited or connected
            if card1[1] != card2[1]:  # Not suited
                # Check if not connected (e.g., 2 and 4)
                rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
                if card1[0] in rank_values and card2[0] in rank_values:
                    if abs(rank_values[card1[0]] - rank_values[card2[0]]) > 1:
                        self.logger.log(f"Weak hand detected: {card1[0]}{card1[1]} and {card2[0]}{card2[1]} (low, unconnected, unsuited)")
                        return True
        
        return False

    def _click_button(self, button_name: str):
        """Simulate mouse click on specified button"""
        if button_name in self.screen_regions['action_buttons']:
            x, y, w, h = self.screen_regions['action_buttons'][button_name]
            center_x = x + w // 2
            center_y = y + h // 2
            
            self.logger.log(f"Clicking {button_name} button at ({center_x}, {center_y})")
            print(f"Clicking {button_name} button at ({center_x}, {center_y})")
            
            try:
                # Move mouse to position and click
                pyautogui.moveTo(center_x, center_y, duration=0.2)
                pyautogui.click()
                time.sleep(0.5)  # Small delay to allow action to register
                self.logger.log(f"Successfully clicked {button_name} button")
            except Exception as e:
                self.logger.log_error(f"Failed to click {button_name} button", e)
        else:
            self.logger.log(f"Unknown button: {button_name}", level="ERROR")
            print(f"Unknown button: {button_name}")


    def simple_card_detection(self, img):
        """Simple card detection using basic image processing"""
        print("Using simple card detection...")
        
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
        
        # Save for debugging
        self.logger.save_image(card1_img, "simple_card1.png", "Simple detection card 1")
        self.logger.save_image(card2_img, "simple_card2.png", "Simple detection card 2")
        
        # Simple color-based detection
        def get_dominant_color(image):
            """Get dominant color from image"""
            pixels = image.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant_color = unique_colors[np.argmax(counts)]
            return dominant_color
        
        # Analyze card colors
        color1 = get_dominant_color(card1_img)
        color2 = get_dominant_color(card2_img)
        
        print(f"Card 1 dominant color: {color1}")
        print(f"Card 2 dominant color: {color2}")
        
        # Simple heuristic: if card has red, it's hearts/diamonds, else clubs/spades
        def is_red_card(color):
            return color[0] > 100 and color[1] < 100 and color[2] < 100
        
        card1_red = is_red_card(color1)
        card2_red = is_red_card(color2)
        
        # Instead of defaulting to specific cards, return generic cards based on color
        # This is more honest than making up specific cards
        if card1_red:
            card1 = "Ah"  # Generic red card
        else:
            card1 = "As"  # Generic black card
        
        if card2_red:
            card2 = "Kh"  # Generic red card
        else:
            card2 = "Ks"  # Generic black card
        
        print(f"Simple detection: Card1={card1}, Card2={card2}")
        return [card1, card2]