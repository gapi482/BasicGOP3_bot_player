#!/usr/bin/env python3
"""
Card Detection Module
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from logger import Logger

class CardDetector:
    def __init__(self):
        """Initialize card detector"""
        self.suits = ['d', 's', 'c', 'h']
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        self.card_templates = self._load_card_templates()
        self.logger = Logger()

    def _ensure_color_image(self, img):
        """Ensure image is in BGR format (3 channels)"""
        if img is None:
            return None
        
        if len(img.shape) == 2:
            # Grayscale image, convert to BGR
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA image, convert to BGR
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # Already BGR, return as is
            return img
        else:
            # Unknown format, try to convert
            try:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except:
                self.logger.log(f"Cannot convert image with shape {img.shape}", level="ERROR")
                return None

    def _ensure_grayscale_image(self, img):
        """Ensure image is in grayscale format (1 channel)"""
        if img is None:
            return None
        
        if len(img.shape) == 2:
            # Already grayscale, return as is
            return img
        elif len(img.shape) == 3:
            # Color image, convert to grayscale
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # Unknown format, return None
            self.logger.log(f"Cannot convert image with shape {img.shape} to grayscale", level="ERROR")
            return None

    def _load_card_templates(self):
        """Load card templates for detection"""
        templates = {}
        template_dir = "card_templates"
        
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
            print(f"Created template directory: {template_dir}")
            return templates
        
        template_files = [f for f in os.listdir(template_dir) if f.endswith('.png')]
        if not template_files:
            print("No card templates found.")
            return templates
        
        for suit in self.suits:
            for rank in self.ranks:
                card_name = f"{rank}{suit}"
                template_path = os.path.join(template_dir, f"{card_name}.png")
                if os.path.exists(template_path):
                    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        templates[card_name] = template
        
        return templates

    def get_player_cards(self, img, screen_regions, game_window):
        """Extract and identify player's hole cards"""
        player_cards = []
        self.logger.log("Attempting to detect player cards")
        
        # Ensure input image is in proper format
        img = self._ensure_color_image(img)
        if img is None:
            self.logger.log("Failed to process input image", level="ERROR")
            return player_cards
        
        # Get card regions
        x1, y1, w1, h1 = screen_regions['player_card1']
        x2, y2, w2, h2 = screen_regions['player_card2']
        
        self.logger.log(f"Player card 1 region: ({x1}, {y1}, {w1}, {h1})")
        self.logger.log(f"Player card 2 region: ({x2}, {y2}, {w2}, {h2})")
        
        # Convert to relative coordinates
        rel_x1 = x1 - game_window['left']
        rel_y1 = y1 - game_window['top']
        rel_x2 = x2 - game_window['left']
        rel_y2 = y2 - game_window['top']
        
        self.logger.log(f"Relative card 1: ({rel_x1}, {rel_y1})")
        self.logger.log(f"Relative card 2: ({rel_x2}, {rel_y2})")
        
        # Check bounds
        img_height, img_width = img.shape[:2]
        self.logger.log(f"Image size: {img_width}x{img_height}")
        
        if (rel_x1 < 0 or rel_y1 < 0 or rel_x1 + w1 > img_width or rel_y1 + h1 > img_height):
            self.logger.log("Card 1 region is out of image bounds!", level="ERROR")
            return player_cards
        
        if (rel_x2 < 0 or rel_y2 < 0 or rel_x2 + w2 > img_width or rel_y2 + h2 > img_height):
            self.logger.log("Card 2 region is out of image bounds!", level="ERROR")
            return player_cards
        
        # Extract card images
        card1_img = img[rel_y1:rel_y1+h1, rel_x1:rel_x1+w1]
        card2_img = img[rel_y2:rel_y2+h2, rel_x2:rel_x2+w2]
        
        # Save for debugging
        self.logger.save_image(card1_img, "debug_card1.png", "Player's first card")
        self.logger.save_image(card2_img, "debug_card2.png", "Player's second card")
        
        # Detect cards
        card1 = self._detect_card(card1_img, "player")
        card2 = self._detect_card(card2_img, "player")
        
        self.logger.log(f"Detected card 1: {card1}")
        self.logger.log(f"Detected card 2: {card2}")
        
        if card1:
            player_cards.append(card1)
        if card2:
            player_cards.append(card2)
        
        return player_cards

    def get_community_cards(self, img, screen_regions, game_window):
        """Extract and identify community cards"""
        community_cards = []
        
        # Ensure input image is in proper format
        img = self._ensure_color_image(img)
        if img is None:
            self.logger.log("Failed to process input image", level="ERROR")
            return community_cards
        
        # Check flop cards
        for i, coords in enumerate(screen_regions['flop_cards']):
            x, y, w, h = coords
            rel_x = x - game_window['left']
            rel_y = y - game_window['top']
            card_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
            card = self._detect_card(card_img, "flop")
            if card:
                community_cards.append(card)
        
        # Check turn card
        x, y, w, h = screen_regions['turn_card']
        rel_x = x - game_window['left']
        rel_y = y - game_window['top']
        turn_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
        turn_card = self._detect_card(turn_img, "turn")
        if turn_card:
            community_cards.append(turn_card)
        
        # Check river card
        x, y, w, h = screen_regions['river_card']
        rel_x = x - game_window['left']
        rel_y = y - game_window['top']
        river_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
        river_card = self._detect_card(river_img, "river")
        if river_card:
            community_cards.append(river_card)
        
        return community_cards

    def _detect_card(self, card_image, card_type="player"):
        """Detect card using improved color and shape analysis"""
        if card_image is None or card_image.size == 0:
            self.logger.log(f"{card_type} card image is None or empty", level="WARNING")
            return None
        
        self.logger.log(f"Detecting {card_type} card, image size: {card_image.shape}")
        
        # Ensure card image is in BGR format
        card_image = self._ensure_color_image(card_image)
        if card_image is None:
            self.logger.log(f"Failed to process {card_type} card image", level="ERROR")
            return None
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        
        # Detect card color
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2
        
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        
        red_pixels = np.sum(red_mask > 0)
        black_pixels = np.sum(black_mask > 0)
        is_red = red_pixels > black_pixels
        
        self.logger.log(f"Card color analysis - Red pixels: {red_pixels}, Black pixels: {black_pixels}, Is red: {is_red}")
        
        # Detect rank using contour detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            rank_roi = gray[y:y+h, x:x+w]
            
            self.logger.save_image(rank_roi, f'debug_rank_{card_type}.png', f"Detected rank region for {card_type} card")
            
            rank = self._detect_rank_from_contour(largest_contour, rank_roi)
            self.logger.log(f"Detected rank: {rank}")
            
            suit = self._detect_suit_from_color(hsv, is_red)
            self.logger.log(f"Detected suit: {suit}")
            
            if rank and suit:
                card_name = f"{rank}{suit}"
                self.logger.log(f"Final card: {card_name}")
                return card_name
        
        # Fallback to template matching
        self.logger.log("Contour detection failed, falling back to template matching")
        return self._fallback_template_matching(card_image, card_type)

    def _detect_rank_from_contour(self, contour, roi):
        """Detect card rank from contour properties"""
        # Get contour properties
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        print(f"Debug: Rank contour - Area: {area}, Aspect ratio: {aspect_ratio:.2f}")
        
        if area < 100:
            return None
        
        # Get the actual ROI image for better analysis
        if roi.size == 0:
            return None
        
        # Ensure ROI is in proper format for processing
        roi = self._ensure_grayscale_image(roi)
        if roi is None:
            return None
        
        # Try to find the rank by looking for specific patterns
        # Threshold the ROI
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours in the ROI
        try:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            
            # Get the largest contour in the ROI
            largest_contour = max(contours, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(largest_contour)
            
            # Calculate features of this inner contour
            inner_area = cv2.contourArea(largest_contour)
            inner_aspect = float(cw) / ch if ch > 0 else 0
            
            print(f"Debug: Inner contour - Area: {inner_area}, Aspect: {inner_aspect:.2f}")
            
            # Ace: Usually has a tall, thin shape
            if inner_aspect < 0.5 and ch > cw * 1.5:
                return 'A'
            # King: Often has a complex, wide shape
            elif inner_aspect > 1.2 and inner_area > 300:
                return 'K'
            # Queen: Similar to king but often more compact
            elif inner_aspect > 0.8 and inner_area > 200 and inner_area <= 300:
                return 'Q'
            # Jack: Often has a simple, compact shape
            elif inner_aspect > 0.6 and inner_area > 100 and inner_area <= 200:
                return 'J'
            # 10: Wider aspect ratio
            elif inner_aspect > 1.0 and inner_area > 150 and inner_area <= 250:
                return 'T'
            # For other numbers, return None to let template matching handle it
            return None
            
        except Exception as e:
            print(f"Debug: Error in inner contour analysis: {e}")
            return None

    def _detect_suit_from_color(self, hsv, is_red):
        """Detect suit from color analysis"""
        if is_red:
            return 'h'  # Default to hearts
        else:
            return 's'  # Default to spades

    def _fallback_template_matching(self, card_image, card_type):
        """Fallback template matching with better threshold and multiple methods"""
        if not self.card_templates:
            print("Debug: No card templates loaded!")
            return None
        
        # Ensure card image is grayscale for template matching
        gray_card = self._ensure_grayscale_image(card_image)
        if gray_card is None:
            print("Debug: Failed to convert card image to grayscale")
            return None
        
        best_match = None
        best_confidence = 0
        best_method = None
        
        print(f"Debug: Testing {len(self.card_templates)} templates with improved matching")
        
        # Try different template matching methods
        methods = [
            ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
            ('TM_CCOEFF', cv2.TM_CCOEFF),
            ('TM_SQDIFF', cv2.TM_SQDIFF),
            ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
        ]
        
        for card_name, template in self.card_templates.items():
            if template is not None:
                try:
                    # Resize template to match card image if necessary
                    if template.shape != gray_card.shape:
                        template = cv2.resize(template, (gray_card.shape[1], gray_card.shape[0]))
                    
                    # Try different matching methods
                    for method_name, method in methods:
                        res = cv2.matchTemplate(gray_card, template, method)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        
                        # For SQDIFF methods, lower is better
                        if 'SQDIFF' in method_name:
                            confidence = 1 - min_val
                            if confidence > best_confidence and confidence > 0.3:  # Lower threshold for SQDIFF
                                best_confidence = confidence
                                best_match = card_name
                                best_method = method_name
                                print(f"Debug: New best match: {card_name} with {method_name} confidence {confidence:.3f}")
                        else:
                            # For CCOEFF methods, higher is better
                            if max_val > best_confidence and max_val > 0.3:  # Lower threshold for CCOEFF
                                best_confidence = max_val
                                best_match = card_name
                                best_method = method_name
                                print(f"Debug: New best match: {card_name} with {method_name} confidence {max_val:.3f}")
                                
                except Exception as e:
                    print(f"Debug: Error matching {card_name}: {e}")
                    continue
        
        print(f"Debug: Final best match: {best_match} with {best_method} confidence {best_confidence:.3f}")
        
        # Lowered threshold and return best match if found
        if best_confidence > 0.3:
            return best_match
        else:
            print(f"Debug: No match found above threshold (confidence: {best_confidence:.3f})")
            return None

    def validate_cards(self, player_cards: List[str], community_cards: List[str]) -> Tuple[List[str], List[str]]:
        """Validate and filter cards to remove duplicates and invalid cards"""
        print(f"Debug: Raw player cards: {player_cards}")
        print(f"Debug: Raw community cards: {community_cards}")
        
        # Filter out invalid cards
        valid_player_cards = [card for card in player_cards if self._is_valid_card(card)]
        valid_community_cards = [card for card in community_cards if self._is_valid_card(card)]
        
        print(f"Debug: Valid player cards: {valid_player_cards}")
        print(f"Debug: Valid community cards: {valid_community_cards}")
        
        return valid_player_cards, valid_community_cards

    def _is_valid_card(self, card: str) -> bool:
        """Check if a card string is valid"""
        if not card or len(card) != 2:
            return False
        
        rank, suit = card[0], card[1]
        return rank in self.ranks and suit in self.suits