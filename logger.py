import os
import datetime
import cv2
import json
from typing import Any, Dict, List

class Logger:
    def __init__(self, base_dir="logs"):
        """Initialize logger with base directory"""
        self.base_dir = base_dir
        self.ensure_directory_exists(base_dir)
        
        # Create session directory
        self.session_dir = self.create_session_directory()
        
        # Create subdirectories
        self.image_dir = os.path.join(self.session_dir, "images")
        self.log_dir = os.path.join(self.session_dir, "logs")
        self.ensure_directory_exists(self.image_dir)
        self.ensure_directory_exists(self.log_dir)
        
        # Initialize log file
        self.log_file_path = os.path.join(self.log_dir, "session.log")
        self.initialize_log_file()
    
    def ensure_directory_exists(self, directory: str):
        """Create directory if it doesn't exist"""
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def create_session_directory(self) -> str:
        """Create a session directory with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.base_dir, f"session_{timestamp}")
        self.ensure_directory_exists(session_dir)
        return session_dir
    
    def initialize_log_file(self):
        """Initialize log file with header"""
        header = f"=== Governor of Poker Bot Session ===\n"
        header += f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def log(self, message: str, level: str = "INFO", console: bool = True):
        """Log a message to file and optionally console"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # Write to log file
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # Print to console if requested
        if console:
            print(message)
    
    def log_card_detection(self, player_cards: List[str], community_cards: List[str]):
        """Log card detection results"""
        self.log(f"Player cards detected: {player_cards}")
        self.log(f"Community cards detected: {community_cards}")
    
    def log_simulation_results(self, win_prob: float, lose_prob: float, tie_prob: float):
        """Log simulation results"""
        self.log(f"Simulation results - Win: {win_prob:.2%}, Lose: {lose_prob:.2%}, Tie: {tie_prob:.2%}")
    
    def log_decision(self, decision: str):
        """Log bot decision"""
        self.log(f"Bot decision: {decision}")
    
    def log_error(self, error: str, exception: Exception = None):
        """Log error message"""
        self.log(f"ERROR: {error}", level="ERROR")
        if exception:
            self.log(f"Exception details: {str(exception)}", level="ERROR")
    
    def save_image(self, image, filename: str, description: str = ""):
        """Save image to image directory"""
        if image is None:
            self.log(f"Failed to save image {filename}: image is None", level="WARNING")
            return
        
        filepath = os.path.join(self.image_dir, filename)
        success = cv2.imwrite(filepath, image)
        
        if success:
            self.log(f"Saved image: {filename} - {description}")
        else:
            self.log(f"Failed to save image: {filename}", level="ERROR")
    
    def save_calibration_data(self, calibration_data: Dict[str, Any]):
        """Save calibration data to session directory"""
        filepath = os.path.join(self.session_dir, "calibration.json")
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        self.log(f"Saved calibration data to {filepath}")
    
    def get_session_directory(self) -> str:
        """Get the current session directory"""
        return self.session_dir