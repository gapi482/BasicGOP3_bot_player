#!/usr/bin/env python3
"""
Main Bot Class for Governor of Poker - Updated with Improved Card Detection
"""
import random
import cv2
import pyautogui
import time
from card_detection import CardDetector
from game_simulation import GameSimulator
from utils import WindowDetector, ScreenshotManager
from card_confirmation import CardConfirmationWindow
import config


class GovernorOfPokerBot:
    def __init__(self, calibration_data, logger):
        """Initialize the bot with calibration data"""
        self.calibration_data = calibration_data
        self.game_window = self.calibration_data['game_window']
        self.screen_regions = self.calibration_data['screen_regions']

        self.logger = logger
        self.logger.log("Initializing Governor of Poker Bot with Improved Card Detection")

        # Initialize components
        self.window_detector = WindowDetector()
        self.screenshot_manager = ScreenshotManager(self.game_window)
        self.card_detector = CardDetector()  # Now using improved template matching
        self.game_simulator = GameSimulator()

        # Safety settings
        pyautogui.PAUSE = 0.5
        pyautogui.FAILSAFE = True

        # Set up callback for a confirmation window to capture fresh screenshots
        self.confirmation_window = CardConfirmationWindow(calibration_data, logger)
        self.confirmation_window.capture_callback = self._capture_fresh_cards

        self.logger.log("Bot initialized successfully with improved card detection")

    def _setup_from_calibration(self):
        """Setup from calibration data"""
        self.game_window = self.calibration_data['game_window']
        self.screen_regions = self.calibration_data['screen_regions']

    def _capture_fresh_cards(self):
        """Capture a fresh screenshot and detect cards - used by the confirmation window"""
        try:
            # Activate the game window
            self.window_detector.activate_game_window()

            # Take screenshot
            screenshot = self.screenshot_manager.capture_game_window()
            if screenshot is None:
                self.logger.log("Failed to capture fresh screenshot", level="WARNING")
                return [], [], {}

            # Detect cards
            player_cards = self.card_detector.get_player_cards(screenshot, self.screen_regions, self.game_window)
            table_cards = self.card_detector.get_table_cards(screenshot, self.screen_regions, self.game_window)

            # Extract card images
            extracted_images = {}
            # Player cards
            for i, region_name in enumerate(['player_card1', 'player_card2']):
                x, y, w, h = self.screen_regions[region_name]
                rel_x = x - self.game_window['left']
                rel_y = y - self.game_window['top']
                card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]
                extracted_images[region_name] = card_img
            # Flop
            for i, coords in enumerate(self.screen_regions['flop_cards']):
                x, y, w, h = coords
                rel_x = x - self.game_window['left']
                rel_y = y - self.game_window['top']
                card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]
                extracted_images[f"flop_{i + 1}"] = card_img
            # Turn and river
            for key in ['turn_card', 'river_card']:
                x, y, w, h = self.screen_regions[key]
                rel_x = x - self.game_window['left']
                rel_y = y - self.game_window['top']
                card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]
                extracted_images[key] = card_img

            if config.PERFORMANCE.get('verbose_logging', False):
                self.logger.log(f"Fresh capture: {len(player_cards)} player cards, {len(table_cards)} table cards")
            return player_cards, table_cards, extracted_images

        except Exception as e:
            self.logger.log_error(f"Error capturing fresh cards: {e}", e)
            return [], [], {}

    def test_card_detection(self):
        """Test the improved card detection system on the actual game screen"""
        # Activate the game window first
        self.window_detector.activate_game_window()

        # Take a screenshot of the game window
        screenshot = self.screenshot_manager.capture_game_window()
        if screenshot is None:
            self.logger.log("Failed to capture screen for card detection test", level="ERROR")
            return

        # Save the screenshot for debugging
        if config.PERFORMANCE.get('save_screenshots', False):
            cv2.imwrite('game_screenshot_test.png', screenshot)

        # Test player cards
        player_cards = self.card_detector.get_player_cards(screenshot, self.screen_regions, self.game_window)

        # Test table cards
        table_cards = self.card_detector.get_table_cards(screenshot, self.screen_regions, self.game_window)

        # Display results
        print(f"\n=== Card Detection Test Results ===")
        print(f"Player Cards: {player_cards}")
        print(f"table Cards: {table_cards}")
        print(f"Total Cards Detected: {len(player_cards) + len(table_cards)}")

        # Detailed analysis of each card region
        print(f"\n=== Detailed Card Analysis ===")

        # Analyze player cards
        for i, region_name in enumerate(['player_card1', 'player_card2']):
            x, y, w, h = self.screen_regions[region_name]
            rel_x = x - self.game_window['left']
            rel_y = y - self.game_window['top']

            # Extract card image
            card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]

            # Save extracted card image for debugging
            if config.PERFORMANCE.get('save_extracted_images', False):
                cv2.imwrite(f'extracted_{region_name}.png', card_img)

            # Check if a card is present
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
                        print(f"    {j + 1}. {template_name}: {score:.3f}")
            else:
                print(f"  Detection skipped - no card present")

            if config.PERFORMANCE.get('save_extracted_images', False):
                print(f"  Image saved as: extracted_{region_name}.png")

        # Analyze table cards
        for i, coords in enumerate(self.screen_regions['flop_cards']):
            x, y, w, h = coords
            rel_x = x - self.game_window['left']
            rel_y = y - self.game_window['top']
            region_name = f"flop_{i + 1}"

            # Extract card image
            card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]

            # Save extracted card image for debugging
            if config.PERFORMANCE.get('save_extracted_images', False):
                cv2.imwrite(f'extracted_{region_name}.png', card_img)

            # Check if a card is present
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
                        print(f"    {j + 1}. {template_name}: {score:.3f}")
            else:
                print(f"  Detection skipped - no card present")

            if config.PERFORMANCE.get('save_extracted_images', False):
                print(f"  Image saved as: extracted_{region_name}.png")

        # Analyze turn card
        x, y, w, h = self.screen_regions['turn_card']
        rel_x = x - self.game_window['left']
        rel_y = y - self.game_window['top']
        region_name = "turn"

        # Extract card image
        card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]

        # Save extracted card image for debugging
        cv2.imwrite(f'extracted_{region_name}.png', card_img)

        # Check if a card is present
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
                    print(f"    {j + 1}. {template_name}: {score:.3f}")
        else:
            print(f"  Detection skipped - no card present")

        print(f"  Image saved as: extracted_{region_name}.png")

        # Analyze river card
        x, y, w, h = self.screen_regions['river_card']
        rel_x = x - self.game_window['left']
        rel_y = y - self.game_window['top']
        region_name = "river"

        # Extract card image
        card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]

        # Save extracted card image for debugging
        cv2.imwrite(f'extracted_{region_name}.png', card_img)

        # Check if a card is present
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
                    print(f"    {j + 1}. {template_name}: {score:.3f}")
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
            card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]
            is_present = self.card_detector._is_card_present(card_img)

            # Use green if a card is present, red if not
            color = (0, 255, 0) if is_present else (0, 0, 255)

            cv2.rectangle(test_img, (rel_x, rel_y), (rel_x + w, rel_y + h), color, 2)

            # Add label
            matched = player_cards[i] if i < len(player_cards) else "None"
            status = "Present" if is_present else "Absent"
            cv2.putText(test_img, f"{region_name}: {matched} ({status})",
                        (rel_x, rel_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw table card regions
        for i, coords in enumerate(self.screen_regions['flop_cards']):
            x, y, w, h = coords
            rel_x = x - self.game_window['left']
            rel_y = y - self.game_window['top']
            region_name = f"flop_{i + 1}"

            # Extract card image to check if present
            card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]
            is_present = self.card_detector._is_card_present(card_img)

            # Use blue if a card is present, red if not
            color = (255, 0, 0) if is_present else (0, 0, 255)

            cv2.rectangle(test_img, (rel_x, rel_y), (rel_x + w, rel_y + h), color, 2)

            # Add label
            matched = table_cards[i] if i < len(table_cards) else "None"
            status = "Present" if is_present else "Absent"
            cv2.putText(test_img, f"{region_name}: {matched} ({status})",
                        (rel_x, rel_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw turn and river
        for i, region_name in enumerate(['turn_card', 'river_card']):
            x, y, w, h = self.screen_regions[region_name]
            rel_x = x - self.game_window['left']
            rel_y = y - self.game_window['top']

            # Extract card image to check if present
            card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]
            is_present = self.card_detector._is_card_present(card_img)

            # Use blue if a card is present, red if not
            color = (255, 0, 0) if is_present else (0, 0, 255)

            cv2.rectangle(test_img, (rel_x, rel_y), (rel_x + w, rel_y + h), color, 2)

            # Add label
            matched = table_cards[3 + i] if (3 + i) < len(table_cards) else "None"
            status = "Present" if is_present else "Absent"
            label = "turn" if region_name == 'turn_card' else "river"
            cv2.putText(test_img, f"{label}: {matched} ({status})",
                        (rel_x, rel_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save the annotated image
        if config.PERFORMANCE.get('save_screenshots', False):
            cv2.imwrite('card_detection_test_results.png', test_img)
            print(f"\n=== Visual Test Results ===")
            print(f"Annotated screenshot saved as: card_detection_test_results.png")
            print(f"Open this file to see the detected cards and their presence status")

        # Summary
        print(f"\n=== Summary ===")
        print(f"Total player cards detected: {len(player_cards)}/2")
        print(f"Total table cards detected: {len(table_cards)}/5")
        print(f"Total cards detected: {len(player_cards) + len(table_cards)}/7")
        if config.PERFORMANCE.get('save_screenshots', False) or config.PERFORMANCE.get('save_extracted_images', False):
            print(f"\nGenerated files:")
            if config.PERFORMANCE.get('save_screenshots', False):
                print(f"  - game_screenshot_test.png (full screenshot)")
                print(f"  - card_detection_test_results.png (with detection results)")
            if config.PERFORMANCE.get('save_extracted_images', False):
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
        table_cards = self.card_detector.get_table_cards(screenshot, self.screen_regions, self.game_window)

        # Prepare extracted images for the confirmation UI (dummy fill support)
        extracted_images = {}
        try:
            # Player cards
            for i, region_name in enumerate(['player_card1', 'player_card2']):
                x, y, w, h = self.screen_regions[region_name]
                rel_x = x - self.game_window['left']
                rel_y = y - self.game_window['top']
                card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]
                extracted_images[region_name] = card_img
            # Flop
            for i, coords in enumerate(self.screen_regions['flop_cards']):
                x, y, w, h = coords
                rel_x = x - self.game_window['left']
                rel_y = y - self.game_window['top']
                card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]
                extracted_images[f"flop_{i + 1}"] = card_img
            # Turn and river
            for key in ['turn_card', 'river_card']:
                x, y, w, h = self.screen_regions[key]
                rel_x = x - self.game_window['left']
                rel_y = y - self.game_window['top']
                card_img = screenshot[rel_y:rel_y + h, rel_x:rel_x + w]
                extracted_images[key] = card_img
        except Exception:
            extracted_images = {}

        # Show the confirmation window if enabled
        if config.BOT_BEHAVIOR.get('enable_card_confirmation', True):
            # Pass extracted images so the UI can prefill values
            confirmation_result = self.confirmation_window.show_confirmation(player_cards, table_cards,
                                                                             extracted_images)

            if confirmation_result['action'] == 'fold':
                self._take_action('fold')
                return
            elif confirmation_result['action'] == 'skip':
                # Use detected cards as-is
                final_player_cards = player_cards
                final_table_cards = table_cards
            else:
                # Use confirmed/corrected cards
                final_player_cards = confirmation_result['player_cards']
                final_table_cards = confirmation_result['table_cards']
        else:
            # Use detected cards directly without confirmation
            final_player_cards = player_cards
            final_table_cards = table_cards

        # Decision-making using GameSimulator Monte Carlo
        if len(final_player_cards) == 2:
            try:
                players = config.SIMULATION.get('default_opponents', 2)
                samples = config.SIMULATION.get('monte_carlo_samples', 5000)
                wins, losses, ties = self.game_simulator.monte_carlo_simulation(
                    final_player_cards, final_table_cards, players=players, samples=samples
                )
                win_prob = float(wins) + 0.5 * float(ties)

                # Use thresholds from config
                fold_th = config.BOT_BEHAVIOR.get('fold_threshold', 0.3)
                check_th = config.BOT_BEHAVIOR.get('check_threshold', 0.6)

                if win_prob < fold_th:
                    action_to_take = 'fold'
                elif win_prob < check_th:
                    # Prefer check, fallback to call
                    action_to_take = 'check'
                else:
                    # Prefer raise/bet
                    action_to_take = 'raise'
                
                action_start = time.time()
                self._take_action(action_to_take)
                action_time = time.time() - action_start
                
                if config.PERFORMANCE.get('verbose_logging', False):
                    self.logger.log(f"Action '{action_to_take}' executed in {action_time:.3f}s")
            except Exception:
                # Fallback to simple evaluation if simulation fails
                all_cards = final_player_cards + final_table_cards
                hand_strength = self._evaluate_hand_strength(all_cards)
                if hand_strength > 0.7:
                    self._take_action('raise')
                elif hand_strength > 0.4:
                    self._take_action('call')
                else:
                    self._take_action('fold')
        else:
            self.logger.log("Can't decide - player hands not recognized properly", level="WARNING")
            # self._take_action('fold')

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
            # Click near the center of the button with a slight jitter
            cx = x + w // 2 + random.randint(-5, 5)
            cy = y + h // 2
            try:
                # Ensure the game window is focused
                self.window_detector.activate_game_window()
            except Exception:
                pass
            pyautogui.moveTo(cx, cy, duration=0.15)
            pyautogui.click(cx, cy)
            time.sleep(config.BOT_BEHAVIOR.get('delay_between_actions', 0.5))
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

            # Draw table card regions
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

            for button_name, coords in self.screen_regions['action_buttons'].items():
                print(f"Button: {button_name}, Coordinates: {coords}")
                x, y, w, h = coords
                rel_x = x - self.game_window['left']
                rel_y = y - self.game_window['top']
                cv2.rectangle(test_img, (rel_x, rel_y), (rel_x + w, rel_y + h), (0, 0, 255), 2)

            # Save test image
            if config.PERFORMANCE.get('save_screenshots', False):
                cv2.imwrite('test_regions.png', test_img)
                print("Test regions image saved as test_regions.png")

    def calibrate_screen(self):
        """Calibrate screen regions"""
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
