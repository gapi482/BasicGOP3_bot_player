#!/usr/bin/env python3
"""
Improved Card Detection Module using Advanced Template Matching
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
from logger import Logger
import config

class AdvancedTemplateMatcher:
    def __init__(self, template_dir="card_templates"):
        """Initialize advanced template matcher"""
        self.template_dir = template_dir
        self.logger = Logger()
        
        # Card definitions
        self.ranks = list(config.CARD_RANKS)
        self.suits = list(config.CARD_SUITS)
        
        # Load all templates
        self.templates = self._load_all_templates()
        
        # Template matching methods to try
        self.matching_methods = [
            ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
            ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
            ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
        ]
        
        # Templates loaded successfully

    def _load_all_templates(self) -> Dict[str, np.ndarray]:
        """Load all card templates in BGR color format"""
        templates = {}

        if not os.path.exists(self.template_dir):
            self.logger.log(f"Template directory not found: {self.template_dir}", level="ERROR")
            return templates

        # Load templates in color (BGR) format for accurate matching
        for rank in self.ranks:
            for suit in self.suits:
                card_name = f"{rank}{suit}"
                template_path = os.path.join(self.template_dir, f"{card_name}.png")

                if os.path.exists(template_path):
                    # Load in color mode (BGR)
                    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                    if template is not None:
                        # Store only the original color template for clean matching
                        templates[card_name] = {
                            'original': template
                        }
                    else:
                        self.logger.log(f"Failed to load template: {card_name}", level="WARNING")
                else:
                    self.logger.log(f"Template not found: {template_path}", level="WARNING")
        return templates


    def match_card(self, card_image: np.ndarray, confidence_threshold: float = 0.3) -> Optional[str]:
        """Match a card image against templates using RGB color for better accuracy"""
        if card_image is None or card_image.size == 0:
            return None

        best_match = None
        best_confidence = 0.0

        try:
            # Ensure the card image is in BGR format (OpenCV standard)
            if len(card_image.shape) == 2:
                # If grayscale, convert to BGR
                card_bgr = cv2.cvtColor(card_image, cv2.COLOR_GRAY2BGR)
            else:
                card_bgr = card_image.copy()

            # Resize to a standard size for consistent matching
            target_height, target_width = 80, 60
            card_resized = cv2.resize(card_bgr, (target_width, target_height))

            # Test against all templates using simple, reliable method
            for card_name, template_dict in self.templates.items():
                # Use the original template for best accuracy
                template = template_dict.get('original')
                if template is None:
                    continue

                # Resize template to match card size
                template_resized = cv2.resize(template, (target_width, target_height))

                # Use normalized correlation coefficient for best results
                result = cv2.matchTemplate(card_resized, template_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                # Update best match
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = card_name

            # Log the best match for debugging
            if best_confidence > confidence_threshold:
                self.logger.log(f"Card matched: {best_match} (confidence: {best_confidence:.3f})")
                return best_match
            else:
                self.logger.log(f"No match found (best: {best_match} at {best_confidence:.3f}, threshold: {confidence_threshold})")
                return None

        except Exception as e:
            self.logger.log_error(f"Card matching failed: {e}", e)
            return None


    def test_template_matching(self, test_image_path: str, expected_card: str = None) -> Dict:
        """Test template matching on a single image"""
        if not os.path.exists(test_image_path):
            return {"error": f"Image not found: {test_image_path}"}
        
        # Read test image
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            return {"error": f"Failed to read image: {test_image_path}"}
        
        results = {}
        
        # Test full card matching with different thresholds
        for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
            match = self.match_card(test_image, threshold)
            results[f'threshold_{threshold}'] = match
        
        # Get the best match with lowest threshold
        best_match = self.match_card(test_image, 0.3)
        results['best_match'] = best_match
        
        # Add expected result if provided
        if expected_card:
            results['expected'] = expected_card
            results['best_match_correct'] = best_match == expected_card
        
        return results
    
    def debug_card_matching(self, card_image: np.ndarray, card_name: str = "unknown") -> Dict:
        """Debug card matching by testing against all templates and showing confidence scores"""
        if card_image is None or card_image.size == 0:
            return {"error": "Card image is None or empty"}

        results = {}

        try:
            # Ensure BGR color image
            if len(card_image.shape) == 2:
                card_bgr = cv2.cvtColor(card_image, cv2.COLOR_GRAY2BGR)
            else:
                card_bgr = card_image.copy()

            # Resize to standard size
            target_height, target_width = 80, 60
            card_resized = cv2.resize(card_bgr, (target_width, target_height))

            # Test against all templates
            template_scores = {}
            for template_name, template_dict in self.templates.items():
                template = template_dict.get('original')
                if template is None:
                    continue

                # Resize template to match card size
                template_resized = cv2.resize(template, (target_width, target_height))

                # Calculate match score
                result = cv2.matchTemplate(card_resized, template_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                template_scores[template_name] = max_val

            # Sort by confidence score
            sorted_scores = sorted(template_scores.items(), key=lambda x: x[1], reverse=True)

            results['top_matches'] = sorted_scores[:10]  # Top 10 matches
            results['best_match'] = sorted_scores[0] if sorted_scores else None

        except Exception as e:
            results['error'] = str(e)

        return results


class CardDetector:
    """Main card detection class that uses the advanced template matcher"""
    
    def __init__(self):
        """Initialize card detector with advanced template matching"""
        self.template_matcher = AdvancedTemplateMatcher()
        self.logger = Logger()
        
    def get_player_cards(self, img, screen_regions, game_window):
        """Extract and identify player cards"""
        player_cards = []
        
        # Check first card
        x, y, w, h = screen_regions['player_card1']
        rel_x = x - game_window['left']
        rel_y = y - game_window['top']
        card1_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
        card1 = self._detect_card(card1_img, "player1")
        if card1:
            player_cards.append(card1)
        
        # Check second card
        x, y, w, h = screen_regions['player_card2']
        rel_x = x - game_window['left']
        rel_y = y - game_window['top']
        card2_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
        card2 = self._detect_card(card2_img, "player2")
        if card2:
            player_cards.append(card2)
        
        return player_cards
    
    def get_table_cards(self, img, screen_regions, game_window):
        """Extract and identify table cards following Texas Hold'em rules"""
        table_cards = []
        
        # Check flop cards (first 3 table cards)
        flop_cards_present = 0
        for i, coords in enumerate(screen_regions['flop_cards']):
            x, y, w, h = coords
            rel_x = x - game_window['left']
            rel_y = y - game_window['top']
            card_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
            
            # Check if card is present before attempting detection
            if self._is_card_present(card_img):
                card = self._detect_card(card_img, f"flop{i+1}")
                if card:
                    table_cards.append(card)
                    flop_cards_present += 1
            else:
                self.logger.log(f"Flop card {i+1} not present, skipping detection")
        
        # Only check turn card if all 3 flop cards are present
        if flop_cards_present == 3:
            x, y, w, h = screen_regions['turn_card']
            rel_x = x - game_window['left']
            rel_y = y - game_window['top']
            turn_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
            
            # Check if card is present before attempting detection
            if self._is_card_present(turn_img):
                turn_card = self._detect_card(turn_img, "turn")
                if turn_card:
                    table_cards.append(turn_card)
            else:
                self.logger.log("Turn card not present, skipping detection")
        else:
            self.logger.log("Turn card skipped - flop cards not present yet")
        
        # Only check river card if turn card is present (Texas Hold'em rules)
        if len(table_cards) >= 4:  # Flop (3) + Turn (1) = 4 cards
            x, y, w, h = screen_regions['river_card']
            rel_x = x - game_window['left']
            rel_y = y - game_window['top']
            river_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
            
            # Check if card is present before attempting detection
            if self._is_card_present(river_img):
                river_card = self._detect_card(river_img, "river")
                if river_card:
                    table_cards.append(river_card)
            else:
                self.logger.log("River card not present, skipping detection")
        else:
            self.logger.log("River card skipped - turn card not present yet")
        
        return table_cards
    
    def _is_card_present(self, card_image: np.ndarray) -> bool:
        """
        Check if a card is actually present in the image region.
        This prevents false positives when cards haven't been dealt yet.
        """
        if card_image is None or card_image.size == 0:
            return False
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Check for white/blank card background
            # A typical card has a white/light background
            mean_intensity = np.mean(gray)
            if mean_intensity > 220:  # Very bright, likely a blank space (increased threshold)
                return False
            
            # Method 2: Check for uniform color (like a green felt table)
            # Calculate standard deviation of pixel values
            std_dev = np.std(gray)
            if std_dev < 5:  # Very uniform, likely not a card (reduced threshold)
                return False
            
            # Method 3: Check for edge density
            # Cards typically have more edges than blank table areas
            edges = cv2.Canny(gray, 50, 150)
            edge_percentage = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])
            if edge_percentage < 0.01:  # Very few edges, likely not a card (reduced threshold)
                return False
            
            # Method 4: Check for specific color patterns
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
            
            # Check for green felt table color (common in poker games)
            green_lower = np.array([35, 50, 50])
            green_upper = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            green_percentage = np.count_nonzero(green_mask) / (hsv.shape[0] * hsv.shape[1])
            
            # If more than 80% of the area is green, it's likely table, not a card (increased threshold)
            if green_percentage > 0.8:
                return False
            
            # If we passed all checks, likely a card is present
            return True
            
        except Exception as e:
            self.logger.log(f"Error in card presence detection: {e}", level="WARNING")
            # Default to assuming card is present if we can't determine
            return True
    
    def _detect_card(self, card_image, card_type="unknown"):
        """Detect card using simplified template matching"""
        if card_image is None or card_image.size == 0:
            self.logger.log(f"{card_type} card image is None or empty", level="WARNING")
            return None
        
        try:
            # Use simplified template matcher with lower threshold for better detection
            matched_card = self.template_matcher.match_card(card_image, confidence_threshold=0.3)
            
            if matched_card:
                return matched_card
            else:
                return None
                
        except Exception as e:
            self.logger.log_error(f"Error detecting {card_type} card: {e}", e)
            return None
    
    def test_card_detection(self, test_image_path, expected_card=None):
        """Test card detection on a single image"""
        return self.template_matcher.test_template_matching(test_image_path, expected_card)
    
    def debug_card_detection(self, card_image, card_type="unknown"):
        """Debug card detection by showing all template match scores"""
        if card_image is None or card_image.size == 0:
            return {"error": f"{card_type} card image is None or empty"}
        
        return self.template_matcher.debug_card_matching(card_image, card_type)