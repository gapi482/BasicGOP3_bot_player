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
        """Load all card templates"""
        templates = {}
        
        if not os.path.exists(self.template_dir):
            self.logger.log(f"Template directory not found: {self.template_dir}", level="ERROR")
            return templates
        
        # Load rank and suit templates separately for better accuracy
        for rank in self.ranks:
            for suit in self.suits:
                card_name = f"{rank}{suit}"
                template_path = os.path.join(self.template_dir, f"{card_name}.png")
                
                if os.path.exists(template_path):
                    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                    if template is not None:
                        # Store multiple versions of the template
                        templates[card_name] = {
                            'original': template,
                            'grayscale': cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),
                            'edges': self._detect_edges(template),
                            'contours': self._extract_main_contour(template)
                        }
                        # Template loaded successfully
                    else:
                        self.logger.log(f"Failed to load template: {card_name}", level="WARNING")
                else:
                    self.logger.log(f"Template not found: {template_path}", level="WARNING")
        
        return templates

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges in template for better matching"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges

    def _extract_main_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract main contour from template"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            main_contour = max(contours, key=cv2.contourArea)
            # Create a mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [main_contour], -1, 255, -1)
            return mask
        return None

    def _multi_method_template_match(self, source_img: np.ndarray, template: np.ndarray) -> Dict[str, float]:
        """Perform template matching with multiple methods"""
        results = {}
        
        # Ensure both images are the same size
        if source_img.shape != template.shape:
            template = cv2.resize(template, (source_img.shape[1], source_img.shape[0]))
        
        # Ensure both images have the same number of channels and data type
        if len(source_img.shape) != len(template.shape):
            # Convert to grayscale if dimensions don't match
            if len(source_img.shape) == 3:
                source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Ensure same data type
        if source_img.dtype != template.dtype:
            template = template.astype(source_img.dtype)
        
        for method_name, method in self.matching_methods:
            try:
                result = cv2.matchTemplate(source_img, template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if method_name in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:
                    confidence = 1 - min_val  # Invert for SQDIFF methods
                else:
                    confidence = max_val
                
                results[method_name] = confidence
                
            except Exception as e:
                self.logger.log(f"Template matching failed for {method_name}: {e}", level="WARNING")
                results[method_name] = 0.0
        
        return results

    def _enhance_image_for_matching(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Enhance image for better template matching"""
        enhanced = {}
        
        try:
            # Original
            enhanced['original'] = image
            
            # Grayscale
            enhanced['grayscale'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhanced contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced['contrast'] = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
            
            # Edges
            enhanced['edges'] = cv2.Canny(enhanced['grayscale'], 50, 150)
            
            # Blurred (for noise reduction)
            enhanced['blurred'] = cv2.GaussianBlur(image, (3, 3), 0)
            
        except Exception as e:
            self.logger.log(f"Image enhancement failed: {e}", level="WARNING")
            enhanced['original'] = image  # Fallback to original
        
        return enhanced

    def match_card(self, card_image: np.ndarray, confidence_threshold: float = 0.3) -> Optional[str]:
        """Match a card image against templates with simplified, reliable approach"""
        if card_image is None or card_image.size == 0:
            return None
        
        best_match = None
        best_confidence = 0.0
        
        try:
            # Convert card image to grayscale for consistent matching
            if len(card_image.shape) == 3:
                card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
            else:
                card_gray = card_image.copy()
            
            # Test against all templates using simple, reliable method
            for card_name, template_dict in self.templates.items():
                # Use the original template for best accuracy
                template = template_dict.get('original')
                if template is None:
                            continue
                        
                # Convert template to grayscale
                if len(template.shape) == 3:
                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                else:
                    template_gray = template.copy()
                
                # Resize template to match card image size
                if card_gray.shape != template_gray.shape:
                    template_gray = cv2.resize(template_gray, (card_gray.shape[1], card_gray.shape[0]))
                
                # Use normalized correlation coefficient for best results
                result = cv2.matchTemplate(card_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                # Update best match
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = card_name
            
            # Return match if confidence is above threshold
            if best_confidence > confidence_threshold:
                return best_match
            else:
                return None
                
        except Exception as e:
            self.logger.log_error(f"Card matching failed: {e}", e)
            return None

    def match_rank_and_suit_separately(self, card_image: np.ndarray, confidence_threshold: float = 0.7) -> Tuple[Optional[str], Optional[str]]:
        """Match rank and suit separately for better accuracy"""
        if card_image is None or card_image.size == 0:
            return None, None
        
        try:
            # Split card into rank and suit regions
            height, width = card_image.shape[:2]
            
            # Rank region (top half)
            rank_region = card_image[:height//2, :]
            
            # Suit region (bottom half)
            suit_region = card_image[height//2:, :]
            
            # Match rank
            best_rank = None
            best_rank_confidence = 0.0
            
            for rank in self.ranks:
                for suit in self.suits:
                    card_name = f"{rank}{suit}"
                    if card_name in self.templates:
                        template_dict = self.templates[card_name]
                        rank_template = template_dict['original'][:template_dict['original'].shape[0]//2, :]
                        
                        # Ensure compatible types
                        if len(rank_region.shape) != len(rank_template.shape):
                            if len(rank_region.shape) == 3:
                                rank_region_gray = cv2.cvtColor(rank_region, cv2.COLOR_BGR2GRAY)
                            else:
                                rank_region_gray = rank_region
                            
                            if len(rank_template.shape) == 3:
                                rank_template = cv2.cvtColor(rank_template, cv2.COLOR_BGR2GRAY)
                        else:
                            rank_region_gray = rank_region
                        
                        enhanced_source = self._enhance_image_for_matching(rank_region_gray)
                        match_results = self._multi_method_template_match(enhanced_source['original'], rank_template)
                        confidence = max(match_results.values())
                        
                        if confidence > best_rank_confidence:
                            best_rank_confidence = confidence
                            best_rank = rank
            
            # Match suit
            best_suit = None
            best_suit_confidence = 0.0
            
            for suit in self.suits:
                # Find any template with this suit
                for rank in self.ranks:
                    card_name = f"{rank}{suit}"
                    if card_name in self.templates:
                        template_dict = self.templates[card_name]
                        suit_template = template_dict['original'][template_dict['original'].shape[0]//2:, :]
                        
                        # Ensure compatible types
                        if len(suit_region.shape) != len(suit_template.shape):
                            if len(suit_region.shape) == 3:
                                suit_region_gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)
                            else:
                                suit_region_gray = suit_region
                            
                            if len(suit_template.shape) == 3:
                                suit_template = cv2.cvtColor(suit_template, cv2.COLOR_BGR2GRAY)
                        else:
                            suit_region_gray = suit_region
                        
                        enhanced_source = self._enhance_image_for_matching(suit_region_gray)
                        match_results = self._multi_method_template_match(enhanced_source['original'], suit_template)
                        confidence = max(match_results.values())
                        
                        if confidence > best_suit_confidence:
                            best_suit_confidence = confidence
                            best_suit = suit
            
            # Apply thresholds
            if best_rank_confidence > confidence_threshold and best_suit_confidence > confidence_threshold:
                self.logger.log(f"Matched rank: {best_rank} ({best_rank_confidence:.3f}), suit: {best_suit} ({best_suit_confidence:.3f})")
                return best_rank, best_suit
            else:
                self.logger.log(f"Separate matching failed. Rank: {best_rank} ({best_rank_confidence:.3f}), Suit: {best_suit} ({best_suit_confidence:.3f})")
                return None, None
                
        except Exception as e:
            self.logger.log_error(f"Separate rank/suit matching failed: {e}", e)
            return None, None

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
        
        # Convert to grayscale
        if len(card_image.shape) == 3:
            card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        else:
            card_gray = card_image.copy()
        
        # Test against all templates
        template_scores = {}
        for template_name, template_dict in self.templates.items():
            template = template_dict.get('original')
            if template is None:
                continue
            
            # Convert template to grayscale
            if len(template.shape) == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template.copy()
            
            # Resize template to match card image size
            if card_gray.shape != template_gray.shape:
                template_gray = cv2.resize(template_gray, (card_gray.shape[1], card_gray.shape[0]))
            
            # Calculate match score
            result = cv2.matchTemplate(card_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            template_scores[template_name] = max_val
        
        # Sort by confidence score
        sorted_scores = sorted(template_scores.items(), key=lambda x: x[1], reverse=True)
        
        results['top_matches'] = sorted_scores[:10]  # Top 10 matches
        results['best_match'] = sorted_scores[0] if sorted_scores else None
        
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
        
        # Only check turn card if flop cards are present (Texas Hold'em rules)
        if flop_cards_present > 0:
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