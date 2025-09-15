#!/usr/bin/env python3
"""
Quick Template Generator
"""
import cv2
import numpy as np
import pyautogui
import os
import time

def quick_template_capture():
    """Quick and dirty template capture"""
    
    # Create directory
    if not os.path.exists("card_templates"):
        os.makedirs("card_templates")
    
    print("Quick Template Capture")
    print("Press Ctrl+C to stop")
    
    card_count = 0
    
    try:
        while True:
            print(f"\nCapture template #{card_count + 1}")
            print("Position the card and press Enter...")
            input()
            
            # Capture screen
            screenshot = pyautogui.screenshot()
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Define card region (adjust these coordinates)
            x, y, w, h = 899, 633, 60, 80
            card_img = img[y:y+h, x:x+w]
            
            # Save template
            template_path = f"card_templates/card_{card_count + 1}.png"
            cv2.imwrite(template_path, card_img)
            
            print(f"Saved: {template_path}")
            card_count += 1
            
            # Small delay
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\nCaptured {card_count} templates")
        print("Now rename the files properly (e.g., Ah.png, Ks.png, etc.)")

if __name__ == "__main__":
    quick_template_capture()