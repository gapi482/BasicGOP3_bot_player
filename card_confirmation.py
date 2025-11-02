#!/usr/bin/env python3
"""
Fast Card Confirmation Window for Poker Bot
"""
import json
import os
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import config
from card_detection import CardDetector


class CardConfirmationWindow:
    def __init__(self, calibration_data, logger):
        self.root = None
        self.window_thread = None
        self.confirmed_cards = None
        self.player_cards = []
        self.table_cards = []
        self.extracted_images = {}
        self.template_images = {}
        self.result = None

        # Themed colors inspired by GOP3
        # Poker green theme
        self.theme_bg = '#0b5d1e'  # poker table green
        self.theme_panel = '#0e6f26'  # slightly lighter/darker green panel
        self.theme_accent = '#f0c674'  # gold accent
        self.theme_text = '#e6eef2'  # near-white text

        # Card options
        self.ranks = list(config.CARD_RANKS)
        self.suit_emojis = config.CARD_EMOJIS
        self.suits = list(config.CARD_SUITS)
        self.card_options = [f"{rank}{suit}" for rank in self.ranks for suit in self.suits]

        # Button reference maps to avoid grid lookups
        self.player1_suit_buttons = {}
        self.player2_suit_buttons = {}
        self.table_suit_buttons = []  # list of dicts per table card

        # Calibration and capture/detector placeholders
        self.calibration_data = calibration_data
        self.game_window = self.calibration_data['game_window']
        self.screen_regions = self.calibration_data['screen_regions']
        self.card_detector = CardDetector()
        self.logger = logger

        # Callback function to capture fresh screenshots (set by bot)
        self.capture_callback = None
        # If host app wants to run Tk in main thread, they should call start_confirmation_ui()
        # which will build UI and enter mainloop without blocking CLI inputs.

    def start_confirmation_ui(self):
        """Create the window and start Tk mainloop on the main thread."""
        try:
            if self.root is None or not self.root.winfo_exists():
                self._build_window()
            # Start mainloop if not already running
            self.root.mainloop()
        except Exception:
            pass

    def load_template_images(self):
        """Load template images for display in the UI"""
        try:
            for card in self.card_options:
                template_path = f"card_templates/{card}.png"
                template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template_img is not None:
                    # Convert BGR to RGB for PIL/Tkinter display
                    template_rgb = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
                    self.template_images[card] = ImageTk.PhotoImage(Image.fromarray(template_rgb))
        except Exception as e:
            self.logger.log(f"Error loading template images: {e}", level="ERROR")

    def show_confirmation(self, detected_player_cards, detected_table_cards, extracted_images):
        """Update internal data and return the latest choice; window is managed on main thread."""
        # Update internal snapshots for Update/New Game actions
        self.player_cards = detected_player_cards.copy() if detected_player_cards else []
        self.table_cards = detected_table_cards.copy() if detected_table_cards else []
        self.extracted_images = extracted_images.copy() if extracted_images else {}

        # Debug output
        self.logger.log(
            f"Received: {len(self.player_cards)} player cards, {len(self.table_cards)} table cards, {len(self.extracted_images)} images")
        if self.player_cards:
            self.logger.log(f"Player cards: {self.player_cards}")
        if self.table_cards:
            self.logger.log(f"Table cards: {self.table_cards}")

        # If window exists, populate it with the latest detections
        try:
            if self.root is not None and self.root.winfo_exists():
                player, table = self._detect_from_extracted_or_existing()
                # If we have detected cards or extracted images, always fill ALL fields
                # This ensures fresh detections overwrite previous values
                has_new_data = (len(player) > 0 or len(table) > 0 or
                                (detected_player_cards and len(detected_player_cards) > 0) or
                                (detected_table_cards and len(detected_table_cards) > 0))

                # Fill all fields if we have new data, otherwise only fill blanks
                # Update UI immediately (we're running Tk on main thread now)
                self._apply_cards_to_ui(player, table, fill_only_blank=not has_new_data)
                self.logger.log(f"UI updated with {len(player)} player, {len(table)} table cards")

                # Force window to update and become visible
                self.root.update_idletasks()
                self.root.update()
                self.root.deiconify()  # Make sure window is not minimized
                self.root.lift()  # Bring to front
        except Exception as e:
            self.logger.log(f"Error prefilling UI: {e}", level="ERROR")

        # Immediately return the most recent user choice if present; otherwise return current detections
        if self.result is not None:
            return dict(self.result)
        return {
            'action': 'skip',
            'player_cards': detected_player_cards or [],
            'table_cards': detected_table_cards or []
        }

    def _build_window(self):
        """Build the confirmation window UI (must be called on main thread)."""
        self.root = tk.Tk()
        self.root.title("Card Confirmation")
        self.root.geometry("780x460")
        self.root.attributes('-topmost', True)  # Always on top
        # Do not force focus; keep CMD usable while this panel stays visible
        try:
            self.root.lift()
        except Exception:
            pass

        # Apply base theme
        self.root.configure(bg=self.theme_bg)

        # Load calibration if available
        self._load_calibration()

        # Load template images (requires an active Tk root)
        self.load_template_images()

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        # Try to theme ttk a bit
        try:
            style = ttk.Style(self.root)
            style.theme_use('clam')
            style.configure('TFrame', background=self.theme_bg)
            style.configure('TLabelframe', background=self.theme_panel, foreground=self.theme_text)
            style.configure('TLabelframe.Label', background=self.theme_panel, foreground=self.theme_text)
            style.configure('TLabel', background=self.theme_bg, foreground=self.theme_text)
            style.configure('Accent.TButton', background=self.theme_accent, foreground='#1b1b1b')
            style.map('Accent.TButton', background=[('active', '#ffd98a')])
            style.configure('TButton', padding=6)
            style.configure('TCombobox', fieldbackground='#ffffff', foreground='#000000')
        except Exception:
            pass

        # Title
        title_label = ttk.Label(main_frame, text="Quick Card Correction",
                                font=("Arial", 12, "bold"))
        title_label.grid(row=0, column=0, columnspan=6, pady=(0, 15))

        # Player cards section
        player_frame = ttk.LabelFrame(main_frame, text="Player Cards", padding="8")
        player_frame.grid(row=1, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))

        # Player card 1
        ttk.Label(player_frame, text="Card 1:").grid(row=0, column=0, padx=(0, 5))
        self.player_card1_rank = tk.StringVar(
            value=self.player_cards[0][0] if len(self.player_cards) > 0 and self.player_cards[0] else "")
        self.player_card1_suit = tk.StringVar(
            value=self.player_cards[0][1] if len(self.player_cards) > 0 and self.player_cards[0] else "")

        # Rank text entry with auto-uppercase
        rank1_entry = ttk.Entry(player_frame, textvariable=self.player_card1_rank, width=3)
        rank1_entry.grid(row=0, column=1, padx=(0, 5))
        self._bind_uppercase(self.player_card1_rank)

        # Suit buttons for card 1
        self.player1_suit_buttons = {}
        for i, (suit, emoji) in enumerate(self.suit_emojis.items()):
            fg = '#c0392b' if suit in ('h', 'd') else '#000000'
            btn = tk.Button(player_frame, text=emoji, width=2, height=1,
                            command=lambda s=suit: self._set_suit(1, s),
                            bg=self._suit_bg(self.player_card1_suit.get(), suit),
                            fg=fg, font=("Arial", 22, "bold"), relief="ridge")
            btn.grid(row=0, column=2 + i, padx=2)
            self.player1_suit_buttons[suit] = btn

        # Player card 2
        ttk.Label(player_frame, text="Card 2:").grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        self.player_card2_rank = tk.StringVar(
            value=self.player_cards[1][0] if len(self.player_cards) > 1 and self.player_cards[1] else "")
        self.player_card2_suit = tk.StringVar(
            value=self.player_cards[1][1] if len(self.player_cards) > 1 and self.player_cards[1] else "")

        rank2_entry = ttk.Entry(player_frame, textvariable=self.player_card2_rank, width=3)
        rank2_entry.grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        self._bind_uppercase(self.player_card2_rank)

        # Suit buttons for card 2
        self.player2_suit_buttons = {}
        for i, (suit, emoji) in enumerate(self.suit_emojis.items()):
            fg = '#c0392b' if suit in ('h', 'd') else '#000000'
            btn = tk.Button(player_frame, text=emoji, width=2, height=1,
                            command=lambda s=suit: self._set_suit(2, s),
                            bg=self._suit_bg(self.player_card2_suit.get(), suit),
                            fg=fg, font=("Arial", 20, "bold"), relief="ridge")
            btn.grid(row=1, column=2 + i, padx=2, pady=(5, 0))
            self.player2_suit_buttons[suit] = btn

        # table cards section
        table_frame = ttk.LabelFrame(main_frame, text="Table Cards", padding="8")
        table_frame.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))

        # table card variables
        self.table_ranks = []
        self.table_suits = []
        self.table_suit_buttons = []
        for i in range(5):
            rank_var = tk.StringVar(
                value=self.table_cards[i][0] if i < len(self.table_cards) and self.table_cards[i] else "")
            suit_var = tk.StringVar(
                value=self.table_cards[i][1] if i < len(self.table_cards) and self.table_cards[i] else "")
            self.table_ranks.append(rank_var)
            self.table_suits.append(suit_var)

            # Row layout: flop (0..2) on row 0, turn/river (3..4) on row 1
            row_idx = 0 if i < 3 else 1
            base_col = (i if i < 3 else (i - 3)) * 6

            ttk.Label(table_frame, text=f"{i + 1}:").grid(row=row_idx, column=base_col, padx=(0, 2), pady=(0, 4))
            rank_entry = ttk.Entry(table_frame, textvariable=rank_var, width=3)
            rank_entry.grid(row=row_idx, column=base_col + 1, padx=(0, 2), pady=(0, 4))
            self._bind_uppercase(rank_var)

            # Suit buttons for table cards
            btn_map = {}
            for j, (suit, emoji) in enumerate(self.suit_emojis.items()):
                fg = '#c0392b' if suit in ('h', 'd') else '#000000'
                btn = tk.Button(table_frame, text=emoji, width=2, height=1,
                                command=lambda s=suit, idx=i: self._set_table_suit(idx, s),
                                bg=self._suit_bg(suit_var.get(), suit),
                                fg=fg, font=("Arial", 20, "bold"), relief="ridge")
                btn.grid(row=row_idx, column=base_col + 2 + j, padx=2, pady=(0, 4))
                btn_map[suit] = btn
            self.table_suit_buttons.append(btn_map)

        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=6, pady=(15, 0))

        # Confirm button
        confirm_btn = ttk.Button(button_frame, text="CONFIRM & PLAY",
                                 command=self._confirm_cards, style='Accent.TButton')
        confirm_btn.grid(row=0, column=0, padx=(0, 10))

        # Fold button
        fold_btn = ttk.Button(button_frame, text="FOLD", command=self._fold_hand)
        fold_btn.grid(row=0, column=1, padx=(0, 10))

        # Skip button
        skip_btn = ttk.Button(button_frame, text="SKIP",
                              command=self._skip_confirmation)
        skip_btn.grid(row=0, column=2)

        # Update State: detect and fill only new cards (blank fields)
        update_btn = ttk.Button(button_frame, text="UPDATE STATE", command=self._update_state)
        update_btn.grid(row=0, column=3, padx=(10, 0))

        # New Game: reset entries and refresh detected player/flop if present
        new_game_btn = ttk.Button(button_frame, text="NEW GAME", command=self._new_game)
        new_game_btn.grid(row=0, column=4, padx=(10, 0))

        # Bind Enter key to confirm
        self.root.bind('<Return>', lambda e: self._confirm_cards())
        self.root.bind('<Escape>', lambda e: self._fold_hand())

        # Focus on first combo box
        try:
            # Focus first rank entry if available
            for child in player_frame.winfo_children():
                if isinstance(child, ttk.Entry):
                    child.focus()
                    break
        except Exception:
            pass

    def _suit_bg(self, current_value, this_suit):
        """Background color for suit buttons based on selection state."""
        return self.theme_accent if current_value == this_suit else '#d0d7de'

    def _bind_uppercase(self, tk_string_var):
        """Bind a trace to always keep the entry uppercase and restricted to valid ranks."""
        valid_set = set(self.ranks)

        def _on_change(*_):
            value = tk_string_var.get().upper()
            # allow T for 10
            if len(value) > 1:
                value = value[0]
            if value and value not in valid_set:
                # if user typed '10', convert to 'T'
                if value == '1':
                    value = 'T'
                else:
                    # keep last valid char only or clear
                    value = value if value in valid_set else ''
            tk_string_var.set(value)

        tk_string_var.trace_add('write', _on_change)

    def _set_suit(self, card_num, suit):
        """Set suit for player card"""
        if card_num == 1:
            self.player_card1_suit.set(suit)
        else:
            self.player_card2_suit.set(suit)
        self._update_suit_buttons()

    def _set_table_suit(self, card_idx, suit):
        """Set suit for table card"""
        self.table_suits[card_idx].set(suit)
        self._update_suit_buttons()

    def _update_suit_buttons(self):
        """Update suit button colors"""
        # Update player card 1 buttons
        for suit, btn in self.player1_suit_buttons.items():
            btn.configure(bg=self._suit_bg(self.player_card1_suit.get(), suit))

        # Update player card 2 buttons
        for suit, btn in self.player2_suit_buttons.items():
            btn.configure(bg=self._suit_bg(self.player_card2_suit.get(), suit))

        # Update table card buttons
        for idx, suit_map in enumerate(self.table_suit_buttons):
            for suit, btn in suit_map.items():
                btn.configure(bg=self._suit_bg(self.table_suits[idx].get(), suit))

    def _apply_cards_to_ui(self, player_cards, table_cards, fill_only_blank=True):
        """Apply given cards into the UI entries. If fill_only_blank is True, only fill empty fields.
        If fill_only_blank is False, also clear fields that don't have corresponding detected cards."""
        # Player 1
        if len(player_cards) >= 1 and player_cards[0]:
            rank = player_cards[0][0].upper()
            suit = player_cards[0][1].lower()
            if not fill_only_blank or self.player_card1_rank.get() == '' or self.player_card1_suit.get() == '':
                self.player_card1_rank.set(rank)
                self.player_card1_suit.set(suit)
        elif not fill_only_blank:
            self.player_card1_rank.set('')
            self.player_card1_suit.set('')

        # Player 2
        if len(player_cards) >= 2 and player_cards[1]:
            rank = player_cards[1][0].upper()
            suit = player_cards[1][1].lower()
            if not fill_only_blank or self.player_card2_rank.get() == '' or self.player_card2_suit.get() == '':
                self.player_card2_rank.set(rank)
                self.player_card2_suit.set(suit)
        elif not fill_only_blank:
            self.player_card2_rank.set('')
            self.player_card2_suit.set('')

        # Table cards (up to 5)
        for i in range(min(5, len(self.table_ranks))):
            if i < len(table_cards) and table_cards[i]:
                rank = table_cards[i][0].upper()
                suit = table_cards[i][1].lower()
                if not fill_only_blank or self.table_ranks[i].get() == '' or self.table_suits[i].get() == '':
                    self.table_ranks[i].set(rank)
                    self.table_suits[i].set(suit)
            elif not fill_only_blank:
                self.table_ranks[i].set('')
                self.table_suits[i].set('')

        # Update suit buttons to reflect the new suit values
        self._update_suit_buttons()

        # Force UI refresh to ensure Entry widgets display updated values
        if self.root and self.root.winfo_exists():
            self.root.update_idletasks()

    def _load_calibration(self):
        """Load the calibration file if present."""
        try:
            calib_path = 'screen_calibration.json'
            if os.path.exists(calib_path):
                with open(calib_path, 'r') as f:
                    data = json.load(f)
                if 'game_window' in data:
                    self.game_window = data['game_window']
                if 'screen_regions' in data:
                    self.screen_regions = data['screen_regions']
                self.logger.log("Calibration loaded successfully")
        except Exception as e:
            self.logger.log(f"Failed to load calibration: {e}", level="ERROR")

    def _detect_from_extracted_or_existing(self):
        """Use already detected values or extracted images to provide suggestions."""
        # Prefer values provided by the bot
        player = list(self.player_cards) if self.player_cards else []
        table = list(self.table_cards) if self.table_cards else []

        # If images are available but values missing, try template matcher quickly
        if self.extracted_images and (len(player) < 2 or len(table) < 5):
            try:
                # Player cards
                for i, key in enumerate(['player_card1', 'player_card2']):
                    if len(player) <= i or not player[i]:
                        img = self.extracted_images.get(key)
                        if img is not None:
                            match = self.card_detector.template_matcher.match_card(img, confidence_threshold=0.3)
                            if match:
                                if len(player) <= i:
                                    player += [None] * (i - len(player) + 1)
                                player[i] = match
                # Flop
                for i in range(3):
                    if len(table) <= i or not table[i]:
                        img = self.extracted_images.get(f"flop_{i + 1}")
                        if img is not None:
                            match = self.card_detector.template_matcher.match_card(img, confidence_threshold=0.3)
                            if match:
                                if len(table) <= i:
                                    table += [None] * (i - len(table) + 1)
                                table[i] = match
                # Turn/River
                for idx, key in enumerate(['turn_card', 'river_card'], start=3):
                    if len(table) <= idx or not table[idx]:
                        img = self.extracted_images.get(key)
                        if img is not None:
                            match = self.card_detector.template_matcher.match_card(img, confidence_threshold=0.3)
                            if match:
                                if len(table) <= idx:
                                    table += [None] * (idx - len(table) + 1)
                                table[idx] = match
            except Exception:
                pass
        return player, table

    def _update_state(self):
        """Fill blank fields by capturing a fresh screenshot and detecting new cards."""
        self.logger.log("Update State button clicked")
        # Try to capture a new screenshot if callback is available
        if self.capture_callback is not None:
            try:
                self.logger.log("Capturing fresh screenshot via callback...")
                fresh_player, fresh_table, fresh_images = self.capture_callback()
                # Update internal state with fresh data
                if fresh_player:
                    self.player_cards = fresh_player
                if fresh_table:
                    self.table_cards = fresh_table
                if fresh_images:
                    self.extracted_images = fresh_images
            except Exception as e:
                self.logger.log(f"Failed to capture fresh screenshot: {e}", level="ERROR")

        # Detect from current state
        player_cards, table_cards = self._detect_from_extracted_or_existing()
        self._apply_cards_to_ui(player_cards, table_cards, fill_only_blank=True)

        # Force UI update
        if self.root and self.root.winfo_exists():
            self.root.update_idletasks()
            self.root.update()

    def _new_game(self):
        """Clear inputs and capture a fresh screenshot for a new hand."""
        self.player_card1_rank.set('')
        self.player_card1_suit.set('')
        self.player_card2_rank.set('')
        self.player_card2_suit.set('')
        for i in range(5):
            self.table_ranks[i].set('')
            self.table_suits[i].set('')
        self._update_suit_buttons()

        # Force UI refresh after clearing
        if self.root and self.root.winfo_exists():
            self.root.update_idletasks()

        # Try to capture a new screenshot if callback is available
        if self.capture_callback is not None:
            try:
                self.logger.log("Capturing fresh screenshot via callback...")
                fresh_player, fresh_table, fresh_images = self.capture_callback()
                # Update internal state with fresh data
                self.player_cards = fresh_player if fresh_player is not None else []
                self.table_cards = fresh_table if fresh_table is not None else []
                self.extracted_images = fresh_images if fresh_images is not None else {}
            except Exception as e:
                self.logger.log(f"Failed to capture fresh screenshot: {e}", level="ERROR")
                self.player_cards = []
                self.table_cards = []
                self.extracted_images = {}
        else:
            # No callback available, clear internal state
            self.player_cards = []
            self.table_cards = []
            self.extracted_images = {}

        # Prefill from detected values
        player_cards, table_cards = self._detect_from_extracted_or_existing()
        self._apply_cards_to_ui(player_cards, table_cards, fill_only_blank=False)
        self.logger.log(f"New Game: Applied {len(player_cards)} player, {len(table_cards)} table cards to UI")

        # Force UI update
        if self.root and self.root.winfo_exists():
            self.root.update_idletasks()
            self.root.update()

    def _confirm_cards(self):
        """Confirm the selected cards"""
        # Get player cards
        player_cards = []
        if self.player_card1_rank.get() and self.player_card1_suit.get():
            player_cards.append(f"{self.player_card1_rank.get()}{self.player_card1_suit.get()}")
        if self.player_card2_rank.get() and self.player_card2_suit.get():
            player_cards.append(f"{self.player_card2_rank.get()}{self.player_card2_suit.get()}")

        # Get table cards
        table_cards = []
        for i in range(5):
            if self.table_ranks[i].get() and self.table_suits[i].get():
                table_cards.append(f"{self.table_ranks[i].get()}{self.table_suits[i].get()}")

        self.result = {
            'action': 'confirm',
            'player_cards': player_cards,
            'table_cards': table_cards
        }

    def _fold_hand(self):
        """Fold the hand"""
        self.result = {
            'action': 'fold',
            'player_cards': [],
            'table_cards': []
        }

    def _skip_confirmation(self):
        """Skip confirmation and use detected cards"""
        self.result = {
            'action': 'skip',
            'player_cards': self.player_cards,
            'table_cards': self.table_cards
        }
