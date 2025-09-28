#!/usr/bin/env python3
"""
Governor of Poker Bot - Main Entry Point
"""

from bot import GovernorOfPokerBot
from card_confirmation import confirm_cards
import sys
from logger import Logger

def main():
    """Main function to run the bot"""
    print("=== Governor of Poker Bot ===")
    
    try:
        # Initialize logger
        logger = Logger()
        logger.log("Starting Governor of Poker Bot application")
        
        # Initialize bot with calibration file
        bot = GovernorOfPokerBot('screen_calibration.json', logger)
        
        # Launch persistent confirmation window immediately (non-blocking)
        try:
            confirm_cards([], [], None)
        except Exception:
            pass
        
        print(f"Game window: {bot.game_window['width']}x{bot.game_window['height']} at ({bot.game_window['left']}, {bot.game_window['top']})")
        print()
        
        while True:
            print("1. Test screen regions")
            print("2. Test card detection")
            print("3. Calibrate screen")
            print("4. Preview game window")
            print("5. Play single hand")
            print("6. Exit")
            choice = input("\nEnter your choice (1-6): ").strip()
            logger.log(f"User selected option: {choice}")
            
            if choice == "1":
                bot.test_regions()
            elif choice == "2":
                bot.test_card_detection()
            elif choice == "3":
                bot.calibrate_screen()
            elif choice == "4":
                bot.preview_game_window()
            elif choice == "5":
                bot.play_hand()
            elif choice == "6":
                logger.log("Exiting application")
                print("Exiting...")
                break
            else:
                logger.log(f"Invalid choice: {choice}", level="WARNING")
                print("Invalid choice. Please try again.")
                
    except KeyboardInterrupt:
        logger.log("Application stopped by user")
        print("\nBot stopped by user")
    except Exception as e:
        logger.log_error(f"Application error: {e}", e)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()