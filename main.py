#!/usr/bin/env python3
"""
Governor of Poker Bot - Main Entry Point
"""

from bot import GovernorOfPokerBot
from card_confirmation import CardConfirmationWindow
import sys
import os
import json
from logger import Logger

def main():
    print("=== Governor of Poker Bot ===")
    logger = Logger()
    logger.log("Starting Governor of Poker Bot application")
    try:
        # Initialize bot with a calibration file
        if os.path.exists("screen_calibration.json"):
            with open("screen_calibration.json", 'r') as f:
                calibration_data = json.load(f)
        else:
            logger.log("No calibration file found, ", level="WARNING")
        bot = GovernorOfPokerBot(calibration_data, logger)
        confirmation_window = CardConfirmationWindow(calibration_data)
        print()

        # Start the card confirmation UI on the main thread so Tk runs safely
        try:
            import threading
            ui_thread = threading.Thread(target=confirmation_window.start_confirmation_ui, daemon=True)
            ui_thread.start()
        except Exception:
            pass

        while True:
            print("1. Play single hand")
            print("2. Test screen regions")
            print("3. Test card detection")
            print("4. Preview game window")
            print("5. Exit")
            choice = input("\nEnter your choice (1-5): ").strip()
            logger.log(f"User selected option: {choice}")

            if choice == "1":
                bot.play_hand()
                try:
                    confirmation_window.show_confirmation([], [], None)
                except Exception:
                    pass
            elif choice == "2":
                bot.test_regions()
            elif choice == "3":
                bot.test_card_detection()
            elif choice == "4":
                bot.preview_game_window()
            elif choice == "5":
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