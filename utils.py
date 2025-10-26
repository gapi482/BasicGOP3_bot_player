#!/usr/bin/env python3
"""
Utility Functions
"""

import cv2
import numpy as np
import pyautogui
import pyscreenshot as ImageGrab
import win32gui
import win32con
import pygetwindow as gw
from typing import Tuple, Dict, Any, Optional
import time
from logger import Logger


class WindowDetector:
    """Detect game window position and size"""
    
    def detect_window(self) -> Dict[str, Any]:
        """Detect the actual game window position and size"""
        print("Detecting game window...")
        
        window_found = False
        game_window = {
            'left': 0,
            'top': 40,
            'width': 1920,
            'height': 1000
        }
        
        hwnd = win32gui.FindWindow(None,"GOP3")
        if hwnd:
            rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = rect
                
            width = right - left
            height = bottom - top
                
            if width > 1000 and height > 600:
                game_window = {
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height
                }
                print(f"Position: ({left}, {top}), Size: {width}x{height}")
                window_found = True
                
        
        if not window_found:
            print("Could not find game window automatically.")
        
        return game_window
    
    def activate_game_window(self):
        """Activate the GOP3 game window to bring it to foreground (gentle like Alt+Tab)"""
        try:
            hwnd = win32gui.FindWindow(None, "GOP3")
            if hwnd:
                # Check if window is minimized
                if win32gui.IsIconic(hwnd):
                    # Only restore if minimized, don't change size if already visible
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    time.sleep(0.2)
                
                # Bring window to foreground without changing dimensions
                win32gui.SetForegroundWindow(hwnd)
                # Wait a moment for window to come to foreground
                time.sleep(0.3)
                return True
            else:
                print("Could not find GOP3 window to activate")
                return False
        except Exception as e:
            print(f"Error activating game window: {e}")
            return False


class ScreenshotManager:
    """Manage screenshot operations"""

    def __init__(self, game_window=None, logger=None):
        self.game_window = game_window
        self.window_capture = GameWindowCapture("GOP3", logger)
        self.logger = logger or Logger()

    def capture_full_screen(self):
        """Capture full screen screenshot"""
        try:
            img = ImageGrab.grab()
            img.save('full_screen_test.png')
            # Read back and ensure proper format
            result = cv2.imread('full_screen_test.png')
            if result is not None:
                result = self.window_capture._ensure_bgr_format(result)
            return result
        except Exception as e:
            self.logger.log_error(f"Error taking full screen screenshot: {e}", e)
            return None

    def capture_game_window(self):
        """Capture game window screenshot"""
        self.logger.log("Taking screenshot...")
        try:
            # Try to use the window capture first
            img = self.window_capture.capture_game_image()

            if img is not None:
                # Save for debugging
                cv2.imwrite('current_game.png', img)
                return img
            else:
                # Fallback to old method
                self.logger.log("Window capture failed, using fallback method")
                im = ImageGrab.grab(bbox=(
                    self.game_window['left'],
                    self.game_window['top'],
                    self.game_window['left'] + self.game_window['width'],
                    self.game_window['top'] + self.game_window['height']
                ))
                screenshot_path = 'current_game.png'
                im.save(screenshot_path)

                # Read back and ensure proper format
                result = cv2.imread(screenshot_path)
                if result is not None:
                    result = self.window_capture._ensure_bgr_format(result)
                return result
        except Exception as e:
            self.logger.log_error(f"Error taking screenshot: {e}", e)
            return None

class ScreenCalibrator:
    """Handle screen calibration"""
    
    def calibrate(self) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
        """Interactive calibration to find exact screen positions"""
        print("=== Screen Calibration Mode ===")
        print("Move your mouse to the specified locations and press ENTER when ready")
        
        # Calibrate window corners
        print("\n=== Step 1: Game Window Calibration ===")
        window_corners = {
            'top_left': "Move mouse to top-left corner of the game window",
            'bottom_right': "Move mouse to bottom-right corner of the game window",
        }
        
        calibrated_window = {}
        
        for corner_name, instruction in window_corners.items():
            print(f"\n{instruction}")
            input("Press ENTER when mouse is in position...")
            
            x, y = pyautogui.position()
            calibrated_window[corner_name] = (x, y)
            print(f"Recorded: {corner_name} at ({x}, {y})")
        
        # Calibrate game elements
        print("\n=== Step 2: Game Elements Calibration ===")
        
        calibration_points = {
            'player_card1': "Move mouse to center of your first (left) hole card",
            'player_card2': "Move mouse to center of your second (right) hole card",
            'flop_card1': "Move mouse to center of first flop card (leftmost)",
            'flop_card3': "Move mouse to center of third flop card (rightmost)",
            'turn_card': "Move mouse to center of turn card position",
            'river_card': "Move mouse to center of river card position",
            'fold_button': "Move mouse to center of FOLD button",
            'check_button': "Move mouse to center of CHECK button",
            'bet_button': "Move mouse to center of BET button",
        }
        
        calibrated_coords = {}
        
        for point_name, instruction in calibration_points.items():
            print(f"\n{instruction}")
            input("Press ENTER when mouse is in position...")
            
            x, y = pyautogui.position()
            calibrated_coords[point_name] = (x, y)
            print(f"Recorded: {point_name} at ({x}, {y})")
        
        return calibrated_coords, calibrated_window

class RegionUpdater:
    """Update screen regions based on calibration"""
    
    def update_regions(self, calibrated_coords: Dict[str, Tuple[int, int]], 
                      screen_regions: Dict[str, Any]) -> Dict[str, Any]:
        """Update screen regions based on calibration points"""
        regions = screen_regions.copy()
        
        # Player cards
        if 'player_card1' in calibrated_coords:
            x, y = calibrated_coords['player_card1']
            regions['player_card1'] = (x-30, y-40, 60, 80)
            
        if 'player_card2' in calibrated_coords:
            x, y = calibrated_coords['player_card2']
            regions['player_card2'] = (x-30, y-40, 60, 80)
        
        # Flop cards
        if 'flop_card1' in calibrated_coords and 'flop_card3' in calibrated_coords:
            x1, y1 = calibrated_coords['flop_card1']
            x3, y3 = calibrated_coords['flop_card3']
            
            regions['flop_cards'][0] = (x1-35, y1-45, 70, 90)
            regions['flop_cards'][2] = (x3-35, y3-45, 70, 90)
            
            # Calculate middle flop card
            x2 = (x1 + x3) // 2 - 35
            regions['flop_cards'][1] = (x2, y1-45, 70, 90)
        
        # Turn and river
        if 'turn_card' in calibrated_coords:
            x, y = calibrated_coords['turn_card']
            regions['turn_card'] = (x-35, y-45, 70, 90)
            
        if 'river_card' in calibrated_coords:
            x, y = calibrated_coords['river_card']
            regions['river_card'] = (x-35, y-45, 70, 90)
        
        # Action buttons
        if 'fold_button' in calibrated_coords:
            x, y = calibrated_coords['fold_button']
            regions['action_buttons']['fold'] = (x-60, y-30, 120, 60)
            
        if 'check_button' in calibrated_coords:
            x, y = calibrated_coords['check_button']
            regions['action_buttons']['check'] = (x-60, y-30, 120, 60)
            
        if 'bet_button' in calibrated_coords:
            x, y = calibrated_coords['bet_button']
            regions['action_buttons']['bet'] = (x-60, y-30, 120, 60)
        
        return regions

class GameWindowCapture:
    """Handle game window detection and image capture"""

    def __init__(self, window_title="GOP3", logger=None):
        self.window_title = window_title
        self.window = None
        self.window_index = 0
        self.last_capture_time = 0
        self.capture_interval = 0.1
        self.logger = logger or Logger()

    def _ensure_bgr_format(self, img):
        """Ensure image is in BGR format"""
        if img is None:
            return None
        
        if len(img.shape) == 2:
            # Grayscale to BGR
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA to BGR
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # Already BGR
            return img
        else:
            self.logger.log(f"Unknown image format: {img.shape}", level="WARNING")
            return img

    def find_window(self, index=0) -> bool:
        """Find game window by title"""
        try:
            # Get all windows with the title
            windows = gw.getWindowsWithTitle(self.window_title)
            self.logger.log(f"Found {len(windows)} windows with title '{self.window_title}'")

            if windows and len(windows) > index:
                self.window = windows[index]
                self.window_index = index
                self.logger.log(f"Selected window {index}: {self.window.title} at {self.window.left},{self.window.top} ({self.window.width}x{self.window.height})")
                return True
            else:
                self.logger.log(f"No windows found with title '{self.window_title}'", level="WARNING")
                return False
        except Exception as e:
            self.logger.log_error(f"Error finding window: {e}", e)
            return False

    def try_all_windows(self) -> bool:
        """Try all windows with the title to find the correct one"""
        try:
            windows = gw.getWindowsWithTitle(self.window_title)
            self.logger.log(f"Found {len(windows)} windows with title '{self.window_title}'")
            
            for i, window in enumerate(windows):
                try:
                    self.logger.log(f"Testing window {i}: {window.title} at {window.left},{window.top} ({window.width}x{window.height})")
                    
                    # Try to activate and capture this window
                    window.activate()
                    time.sleep(0.5)  # Wait for window to come to foreground
                    
                    # Capture a test image
                    img = self.capture_window_image(window)
                    if img is not None:
                        # Save test image for verification
                        test_path = f"window_test_{i}.png"
                        cv2.imwrite(test_path, img)
                        self.logger.log(f"Saved test image for window {i} as {test_path}")
                        
                        # Ask user if this is the correct window
                        response = input(f"Is window {i} the correct game window? (y/n): ").lower()
                        if response == 'y':
                            self.window = window
                            self.window_index = i
                            return True
                except Exception as e:
                    self.logger.log_error(f"Error testing window {i}: {e}", e)
            
            return False
        except Exception as e:
            self.logger.log_error(f"Error finding windows: {e}", e)
            return False
    
    def capture_window_image(self, window=None) -> Optional[np.ndarray]:
        """Capture image from specified window using multiple methods"""
        if window is None:
            window = self.window
            
        if window is None:
            self.logger.log("No window specified", level="WARNING")
            return None
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_capture_time < self.capture_interval:
            time.sleep(self.capture_interval - (current_time - self.last_capture_time))
        self.last_capture_time = time.time()
        
        # Try multiple capture methods
        img = None
        
        # Method 1: Use pyautogui with window bounds (works reliably on Windows)
        try:
            x, y, width, height = window.left, window.top, window.width, window.height
            self.logger.log(f"Attempting capture with pyautogui bounds: {x},{y} {width}x{height}")
            if width <= 0 or height <= 0:
                self.logger.log(f"Invalid window dimensions: {width}x{height}", level="ERROR")
                return None
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            self.logger.log("Successfully captured image using pyautogui")
            
        except Exception as e:
            self.logger.log_error(f"Method 1 (pyautogui) failed: {e}", e)
        
        # Method 2: Use pyscreenshot if method 1 fails
        if img is None:
            try:
                self.logger.log("Attempting capture with pyscreenshot")
                screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
                img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                self.logger.log("Successfully captured image using pyscreenshot")
            except Exception as e:
                self.logger.log_error(f"Method 2 (pyscreenshot) failed: {e}", e)
        
        # Method 3: Use win32gui if available
        if img is None:
            try:
                import win32gui
                import win32ui
                import win32con
                
                self.logger.log("Attempting capture with win32gui")
                
                # Get window device context
                hwnd = window._hWnd
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                width = right - left
                height = bottom - top
                
                # Create device context
                hwndDC = win32gui.GetWindowDC(hwnd)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()
                
                # Create bitmap
                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
                
                # Select bitmap
                saveDC.SelectObject(saveBitMap)
                
                # BitBlt
                result = saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
                
                # Convert to OpenCV format
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                img = np.frombuffer(bmpstr, dtype='uint8')
                img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Cleanup
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwndDC)
                
                self.logger.log("Successfully captured image using win32gui")
                
            except Exception as e:
                self.logger.log_error(f"Method 3 failed: {e}", e)
        
        if img is not None:
            self.logger.log(f"Final image shape: {img.shape}")
            return img
        else:
            self.logger.log("All capture methods failed", level="ERROR")
            return None
    
    def capture_game_image(self) -> Optional[np.ndarray]:
        """Capture image from the game window"""
        return self.capture_window_image()
    
    def activate_window(self) -> bool:
        """Bring game window to foreground"""
        if self.window:
            try:
                self.window.activate()
                time.sleep(0.5)  # Wait for window to come to foreground
                self.logger.log("Window activated successfully")
                return True
            except Exception as e:
                self.logger.log_error(f"Error activating window: {e}", e)
                return False
        return False
    
    def get_window_position(self) -> Optional[Tuple[int, int, int, int]]:
        """Get window position and size"""
        if self.window:
            return (self.window.left, self.window.top, self.window.width, self.window.height)
        return None