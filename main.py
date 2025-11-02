#!/usr/bin/env python3
"""
Governor of Poker Bot - Main Entry Point
"""

from bot import GovernorOfPokerBot
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

        # Use the confirmation window from the bot (it already has the callback set up)
        confirmation_window = bot.confirmation_window
        print()

        # Build the confirmation window (but don't start mainloop - we'll update it manually)
        try:
            confirmation_window._build_window()
            # Process any pending Tk events
            confirmation_window.root.update()
            logger.log("Card confirmation window initialized")
        except Exception as e:
            logger.log(f"Could not initialize confirmation window: {e}", level="ERROR")

        while True:
            # Process pending Tk events to keep window responsive
            try:
                if confirmation_window.root and confirmation_window.root.winfo_exists():
                    confirmation_window.root.update()
            except Exception:
                pass

            print("1. Play single hand")
            print("2. Test screen regions")
            print("3. Test card detection")
            print("4. Preview game window")
            print("5. Exit")
            choice = input("\nEnter your choice (1-5): ").strip()
            logger.log(f"User selected option: {choice}")

            if choice == "1":
                bot.play_hand()
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
