#!/usr/bin/env python3
"""
Advanced Card Detection Module with OCR and Multiple Detection Methods
"""
import cv2
import numpy as np
import os
import pytesseract
from logger import Logger


class AdvancedCardDetector:
    def __init__(self):
        """Initialize advanced card detector"""
        self.suits = ['d', 's', 'c', 'h']
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        self.logger = Logger()
        
        # Initialize Tesseract OCR
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Load templates for fallback
        self.rank_templates = self._load_rank_templates()
        self.suit_templates = self._load_suit_templates()
        
        # Card detection parameters
        self.card_aspect_ratio = 1.4  # Standard playing card aspect ratio
        self.min_card_area = 1000
        self.max_card_area = 50000

    def _load_rank_templates(self):
        """Load rank templates for OCR fallback"""
        templates = {}
        template_dir = "rank_templates"
        if os.path.exists(template_dir):
            for rank in self.ranks:
                template_path = os.path.join(template_dir, f"{rank}.png")
                if os.path.exists(template_path):
                    templates[rank] = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        return templates

    def _load_suit_templates(self):
        """Load suit templates for OCR fallback"""
        templates = {}
        template_dir = "suit_templates"
        if os.path.exists(template_dir):
            for suit in self.suits:
                template_path = os.path.join(template_dir, f"{suit}.png")
                if os.path.exists(template_path):
                    templates[suit] = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        return templates

    def _preprocess_for_ocr(self, card_image):
        """Preprocess card image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better text detection
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Remove noise
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return processed

    def _extract_rank_with_ocr(self, card_image):
        """Extract card rank using OCR"""
        try:
            # Preprocess image
            processed = self._preprocess_for_ocr(card_image)
            
            # Configure Tesseract for single character recognition
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=AKQJT98765432'
            
            # Perform OCR
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            # Clean and validate the result
            text = text.strip().upper()
            
            # Map common OCR errors
            ocr_mapping = {
                '0': 'O',  # Sometimes 0 is detected as O
                'S': '5',  # Sometimes S is detected as 5
            }
            
            for wrong, correct in ocr_mapping.items():
                text = text.replace(wrong, correct)
            
            # Validate rank
            if text in self.ranks:
                return text
            elif len(text) == 1 and text.isdigit() and text in '23456789':
                return text
            else:
                self.logger.log(f"OCR detected invalid rank: {text}", level="WARNING")
                return None
                
        except Exception as e:
            self.logger.log(f"OCR error: {e}", level="ERROR")
            return None

    def _extract_suit_with_ocr(self, card_image):
        """Extract card suit using OCR and color analysis"""
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for red and black suits
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 100, 100])
            red_upper2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = red_mask1 + red_mask2
            
            # Count red pixels
            red_pixels = np.sum(red_mask > 0)
            total_pixels = card_image.shape[0] * card_image.shape[1]
            red_ratio = red_pixels / total_pixels
            
            # Determine if it's a red suit
            is_red = red_ratio > 0.1  # At least 10% red pixels
            
            # Use OCR for suit symbol detection
            processed = self._preprocess_for_ocr(card_image)
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=HSDC'
            text = pytesseract.image_to_string(processed, config=custom_config).strip().upper()
            
            # Map OCR results to suits
            suit_mapping = {
                'H': 'h', '♥': 'h',
                'S': 's', '♠': 's',
                'D': 'd', '♦': 'd',
                'C': 'c', '♣': 'c'
            }
            
            if text in suit_mapping:
                return suit_mapping[text]
            elif is_red:
                return 'h'  # Default to hearts for red
            else:
                return 's'  # Default to spades for black
                
        except Exception as e:
            self.logger.log(f"Suit detection error: {e}", level="ERROR")
            return None

    def _detect_card_with_contours(self, image):
        """Detect cards using contour detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cards = []
        for contour in contours:
            # Get contour properties
            area = cv2.contourArea(contour)
            
            if self.min_card_area < area < self.max_card_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if it matches card aspect ratio
                if 0.6 < aspect_ratio < 0.8:  # Card is typically taller than wide
                    # Extract card region
                    card_roi = image[y:y+h, x:x+w]
                    cards.append((card_roi, (x, y, w, h)))
        
        return cards

    def _fallback_template_matching(self, card_image, card_type="unknown"):
        """Fallback to template matching if OCR fails"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
            
            # Try rank detection
            best_rank = None
            best_rank_confidence = 0
            
            for rank, template in self.rank_templates.items():
                if template is not None:
                    # Resize template to match card image
                    if template.shape != gray.shape:
                        template = cv2.resize(template, (gray.shape[1], gray.shape[0]))
                    
                    # Template matching
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    if max_val > best_rank_confidence and max_val > 0.7:
                        best_rank_confidence = max_val
                        best_rank = rank
            
            # Try suit detection using color
            hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
            red_pixels = np.sum(red_mask > 0) + np.sum(red_mask2 > 0)
            
            is_red = red_pixels > 1000
            best_suit = 'h' if is_red else 's'
            
            if best_rank:
                return f"{best_rank}{best_suit}"
            
        except Exception as e:
            self.logger.log(f"Template matching error: {e}", level="ERROR")
        
        return None

    def detect_cards(self, image, card_type="player"):
        """Main card detection method using multiple approaches"""
        if image is None or image.size == 0:
            self.logger.log(f"{card_type} image is None or empty", level="WARNING")
            return []
        
        # Ensure proper image format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        detected_cards = []
        
        # Method 1: Direct OCR on the whole image
        try:
            rank = self._extract_rank_with_ocr(image)
            suit = self._extract_suit_with_ocr(image)
            
            if rank and suit:
                card = f"{rank}{suit}"
                detected_cards.append(card)
                self.logger.log(f"OCR detected {card_type} card: {card}")
                return detected_cards
        except Exception as e:
            self.logger.log(f"Direct OCR failed: {e}", level="WARNING")
        
        # Method 2: Contour-based detection
        try:
            cards = self._detect_card_with_contours(image)
            for card_roi, (x, y, w, h) in cards:
                rank = self._extract_rank_with_ocr(card_roi)
                suit = self._extract_suit_with_ocr(card_roi)
                
                if rank and suit:
                    card = f"{rank}{suit}"
                    detected_cards.append(card)
                    self.logger.log(f"Contour+OCR detected {card_type} card: {card}")
                else:
                    # Fallback to template matching
                    card = self._fallback_template_matching(card_roi, card_type)
                    if card:
                        detected_cards.append(card)
                        self.logger.log(f"Template matching detected {card_type} card: {card}")
        except Exception as e:
            self.logger.log(f"Contour detection failed: {e}", level="WARNING")
        
        # Method 3: Direct template matching as last resort
        if not detected_cards:
            card = self._fallback_template_matching(image, card_type)
            if card:
                detected_cards.append(card)
                self.logger.log(f"Final fallback detected {card_type} card: {card}")
        
        return detected_cards

    def get_player_cards(self, img, screen_regions, game_window):
        """Extract and identify player's hole cards"""
        player_cards = []
        
        # Get card regions
        x1, y1, w1, h1 = screen_regions['player_card1']
        x2, y2, w2, h2 = screen_regions['player_card2']
        
        # Convert to relative coordinates
        rel_x1 = x1 - game_window['left']
        rel_y1 = y1 - game_window['top']
        rel_x2 = x2 - game_window['left']
        rel_y2 = y2 - game_window['top']
        
        # Extract card images
        card1_img = img[rel_y1:rel_y1+h1, rel_x1:rel_x1+w1]
        card2_img = img[rel_y2:rel_y2+h2, rel_x2:rel_x2+w2]
        
        # Detect cards using advanced methods
        cards1 = self.detect_cards(card1_img, "player_card1")
        cards2 = self.detect_cards(card2_img, "player_card2")
        
        if cards1:
            player_cards.append(cards1[0])
        if cards2:
            player_cards.append(cards2[0])
        
        return player_cards

    def get_community_cards(self, img, screen_regions, game_window):
        """Extract and identify community cards"""
        community_cards = []
        
        # Check flop cards
        for i, coords in enumerate(screen_regions['flop_cards']):
            x, y, w, h = coords
            rel_x = x - game_window['left']
            rel_y = y - game_window['top']
            card_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
            cards = self.detect_cards(card_img, f"flop_card{i}")
            if cards:
                community_cards.append(cards[0])
        
        # Check turn card
        x, y, w, h = screen_regions['turn_card']
        rel_x = x - game_window['left']
        rel_y = y - game_window['top']
        turn_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
        cards = self.detect_cards(turn_img, "turn_card")
        if cards:
            community_cards.append(cards[0])
        
        # Check river card
        x, y, w, h = screen_regions['river_card']
        rel_x = x - game_window['left']
        rel_y = y - game_window['top']
        river_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
        cards = self.detect_cards(river_img, "river_card")
        if cards:
            community_cards.append(cards[0])
        
        return community_cards