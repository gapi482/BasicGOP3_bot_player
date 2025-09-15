#!/usr/bin/env python3
"""
Improved Card Detection Module using Advanced Template Matching
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
from logger import Logger

class AdvancedTemplateMatcher:
    def __init__(self, template_dir="card_templates"):
        """Initialize advanced template matcher"""
        self.template_dir = template_dir
        self.logger = Logger()
        
        # Card definitions
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        self.suits = ['h', 'd', 'c', 's']
        
        # Load all templates
        self.templates = self._load_all_templates()
        
        # Template matching methods to try
        self.matching_methods = [
            ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
            ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
            ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
        ]
        
        print(f"Loaded {len(self.templates)} card templates")

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
                        self.logger.log(f"Loaded template: {card_name}")
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

    def match_card(self, card_image: np.ndarray, confidence_threshold: float = 0.8) -> Optional[str]:
        """Match a card image against templates with improved validation"""
        if card_image is None or card_image.size == 0:
            return None
        
        best_match = None
        best_confidence = 0.0
        best_method = ""
        
        try:
            # Enhance the source image
            enhanced_source = self._enhance_image_for_matching(card_image)
            
            # Test against all templates
            for card_name, template_dict in self.templates.items():
                for source_type, source_img in enhanced_source.items():
                    for template_type, template_img in template_dict.items():
                        if template_img is None:
                            continue
                        
                        # Skip incompatible types
                        if source_type == 'edges' and template_type != 'edges':
                            continue
                        if template_type == 'edges' and source_type != 'edges':
                            continue
                        
                        # Ensure compatible image types
                        if len(source_img.shape) != len(template_img.shape):
                            if len(source_img.shape) == 3:
                                source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
                            if len(template_img.shape) == 3:
                                template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
                        
                        # Perform matching
                        match_results = self._multi_method_template_match(source_img, template_img)
                        
                        # Find best confidence for this template
                        max_confidence = max(match_results.values())
                        best_method_for_template = max(match_results.items(), key=lambda x: x[1])[0]
                        
                        if max_confidence > best_confidence:
                            best_confidence = max_confidence
                            best_match = card_name
                            best_method = f"{source_type}_{template_type}_{best_method_for_template}"
            
            # IMPROVED VALIDATION: More flexible approach
            if best_confidence > confidence_threshold:
                # Try separate rank/suit matching as a secondary validation
                rank, suit = self.match_rank_and_suit_separately(card_image, confidence_threshold * 0.9)  # Slightly higher threshold for separate matching
                
                if rank and suit:
                    combined_match = f"{rank}{suit}"
                    
                    # Case 1: Full match and separate match agree exactly
                    if combined_match == best_match:
                        self.logger.log(f"Validated match (exact): {best_match} with confidence {best_confidence:.3f} using {best_method}")
                        return best_match
                    
                    # Case 2: Full match and separate match partially agree (either rank or suit matches)
                    elif best_match[0] == rank or best_match[1] == suit:
                        # If confidence is very high, trust the full match
                        if best_confidence > 0.9:
                            self.logger.log(f"Validated match (partial): {best_match} with confidence {best_confidence:.3f} using {best_method}")
                            self.logger.log(f"Partial match with separate detection: {combined_match}")
                            return best_match
                        # Otherwise, use the separate match if its confidence is good
                        else:
                            self.logger.log(f"Using separate match: {combined_match} due to partial agreement")
                            return combined_match
                    
                    # Case 3: Full match and separate match disagree completely
                    else:
                        # If full match confidence is very high, trust it
                        if best_confidence > 0.95:
                            self.logger.log(f"High confidence full match: {best_match} with confidence {best_confidence:.3f}")
                            return best_match
                        # Otherwise, no consensus, return None
                        else:
                            self.logger.log(f"Match validation failed. Full: {best_match}, Separate: {combined_match}")
                            return None
                else:
                    # If separate matching failed but full match confidence is very high, trust it
                    if best_confidence > 0.95:
                        self.logger.log(f"High confidence match without separate validation: {best_match} with confidence {best_confidence:.3f}")
                        return best_match
                    else:
                        self.logger.log(f"Separate matching failed for validation of {best_match}")
                        return None
            else:
                self.logger.log(f"No match found above threshold. Best: {best_match} with {best_confidence:.3f}")
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
        
        # Test full card matching
        full_match = self.match_card(test_image)
        results['full_match'] = full_match
        
        # Test separate rank/suit matching
        rank, suit = self.match_rank_and_suit_separately(test_image)
        results['separate_match'] = f"{rank}{suit}" if rank and suit else None
        
        # Test with different confidence thresholds
        for threshold in [0.9, 0.8, 0.7, 0.6]:
            match = self.match_card(test_image, threshold)
            results[f'threshold_{threshold}'] = match
        
        # Add expected result if provided
        if expected_card:
            results['expected'] = expected_card
            results['full_match_correct'] = full_match == expected_card
            if rank and suit:
                separate_result = f"{rank}{suit}"
                results['separate_match_correct'] = separate_result == expected_card
        
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
    
    def get_community_cards(self, img, screen_regions, game_window):
        """Extract and identify community cards"""
        community_cards = []
        
        # Check flop cards
        for i, coords in enumerate(screen_regions['flop_cards']):
            x, y, w, h = coords
            rel_x = x - game_window['left']
            rel_y = y - game_window['top']
            card_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
            
            # Check if card is present before attempting detection
            if self._is_card_present(card_img):
                card = self._detect_card(card_img, f"flop{i+1}")
                if card:
                    community_cards.append(card)
            else:
                self.logger.log(f"Flop card {i+1} not present, skipping detection")
        
        # Check turn card
        x, y, w, h = screen_regions['turn_card']
        rel_x = x - game_window['left']
        rel_y = y - game_window['top']
        turn_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
        
        # Check if card is present before attempting detection
        if self._is_card_present(turn_img):
            turn_card = self._detect_card(turn_img, "turn")
            if turn_card:
                community_cards.append(turn_card)
        else:
            self.logger.log("Turn card not present, skipping detection")
        
        # Check river card
        x, y, w, h = screen_regions['river_card']
        rel_x = x - game_window['left']
        rel_y = y - game_window['top']
        river_img = img[rel_y:rel_y+h, rel_x:rel_x+w]
        
        # Check if card is present before attempting detection
        if self._is_card_present(river_img):
            river_card = self._detect_card(river_img, "river")
            if river_card:
                community_cards.append(river_card)
        else:
            self.logger.log("River card not present, skipping detection")
        
        return community_cards
    
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
            if mean_intensity > 200:  # Very bright, likely a blank space
                return False
            
            # Method 2: Check for uniform color (like a green felt table)
            # Calculate standard deviation of pixel values
            std_dev = np.std(gray)
            if std_dev < 10:  # Very uniform, likely not a card
                return False
            
            # Method 3: Check for edge density
            # Cards typically have more edges than blank table areas
            edges = cv2.Canny(gray, 50, 150)
            edge_percentage = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])
            if edge_percentage < 0.02:  # Very few edges, likely not a card
                return False
            
            # Method 4: Check for specific color patterns
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
            
            # Check for green felt table color (common in poker games)
            green_lower = np.array([35, 50, 50])
            green_upper = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            green_percentage = np.count_nonzero(green_mask) / (hsv.shape[0] * hsv.shape[1])
            
            # If more than 70% of the area is green, it's likely table, not a card
            if green_percentage > 0.7:
                return False
            
            # If we passed all checks, likely a card is present
            return True
            
        except Exception as e:
            self.logger.log(f"Error in card presence detection: {e}", level="WARNING")
            # Default to assuming card is present if we can't determine
            return True
    
    def _detect_card(self, card_image, card_type="unknown"):
        """Detect card using advanced template matching"""
        if card_image is None or card_image.size == 0:
            self.logger.log(f"{card_type} card image is None or empty", level="WARNING")
            return None
        
        self.logger.log(f"Detecting {card_type} card, image size: {card_image.shape}")
        
        try:
            # Use the advanced template matcher
            matched_card = self.template_matcher.match_card(card_image, confidence_threshold=0.7)
            
            if matched_card:
                self.logger.log(f"Successfully detected {card_type} card as: {matched_card}")
                return matched_card
            else:
                self.logger.log(f"Failed to detect {card_type} card", level="WARNING")
                return None
                
        except Exception as e:
            self.logger.log_error(f"Error detecting {card_type} card: {e}", e)
            return None
    
    def test_card_detection(self, test_image_path, expected_card=None):
        """Test card detection on a single image"""
        return self.template_matcher.test_template_matching(test_image_path, expected_card)