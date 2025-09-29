#!/usr/bin/env python3
"""
Fast Card Confirmation Window for Poker Bot
"""
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import config
import threading
import time
from card_detection import CardDetector

class CardConfirmationWindow:
    def __init__(self):
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
        self.theme_bg = '#0b5d1e'       # poker table green
        self.theme_panel = '#0e6f26'    # slightly lighter/darker green panel
        self.theme_accent = '#f0c674'   # gold accent
        self.theme_text = '#e6eef2'     # near-white text
        
        # Card options
        self.ranks = list(config.CARD_RANKS)
        self.suit_emojis = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
        self.suits = list(config.CARD_SUITS)
        self.card_options = [f"{rank}{suit}" for rank in self.ranks for suit in self.suits]
        
        # Button reference maps to avoid grid lookups
        self.player1_suit_buttons = {}
        self.player2_suit_buttons = {}
        self.table_suit_buttons = []  # list of dicts per table card

        # Calibration and capture/detector placeholders
        self.game_window = dict(config.DEFAULT_GAME_WINDOW)
        self.screen_regions  = dict(config.DEFAULT_SCREEN_REGIONS)
        #sr['flop_cards'] = list(config.DEFAULT_SCREEN_REGIONS['flop_cards'])
        #self.screen_regions = sr
        self.card_detector = CardDetector()
        
    def load_template_images(self):
        """Load template images for display"""
        try:
            for card in self.card_options:
                template_path = f"card_templates/{card}.png"
                template_img = cv2.imread(template_path)
                if template_img is not None:
                    # Resize template for display
                    template_resized = cv2.resize(template_img, (60, 80))
                    template_rgb = cv2.cvtColor(template_resized, cv2.COLOR_BGR2RGB)
                    self.template_images[card] = ImageTk.PhotoImage(Image.fromarray(template_rgb))
        except Exception as e:
            print(f"Error loading template images: {e}")
    
    def show_confirmation(self, detected_player_cards, detected_table_cards, extracted_images):
        """Show the card confirmation window"""
        self.player_cards = detected_player_cards.copy() if detected_player_cards else []
        self.table_cards = detected_table_cards.copy() if detected_table_cards else []
        self.extracted_images = extracted_images.copy() if extracted_images else {}

        try:
            if self.root is not None and self.root.winfo_exists():
                try:
                    self.root.lift()
                except Exception:
                    pass
            else:
                # Start the Tk window in its own thread to keep CLI responsive
                def _runner():
                    self._create_window()
                self.window_thread = threading.Thread(target=_runner, daemon=True)
                self.window_thread.start()
        except Exception:
            pass

        # Do not block; return a non-committal result
        return {
            'action': 'skip',
            'player_cards': detected_player_cards or [],
            'table_cards': detected_table_cards or []
        }
    
    def _create_window(self):
        """Create the confirmation window"""
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
        self.player_card1_rank = tk.StringVar(value=self.player_cards[0][0] if len(self.player_cards) > 0 and self.player_cards[0] else "")
        self.player_card1_suit = tk.StringVar(value=self.player_cards[0][1] if len(self.player_cards) > 0 and self.player_cards[0] else "")
        
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
                           fg=fg, font=("Arial", 22, "bold"), relief=tk.RIDGE)
            btn.grid(row=0, column=2+i, padx=2)
            self.player1_suit_buttons[suit] = btn
        
        # Player card 2
        ttk.Label(player_frame, text="Card 2:").grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        self.player_card2_rank = tk.StringVar(value=self.player_cards[1][0] if len(self.player_cards) > 1 and self.player_cards[1] else "")
        self.player_card2_suit = tk.StringVar(value=self.player_cards[1][1] if len(self.player_cards) > 1 and self.player_cards[1] else "")
        
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
                           fg=fg, font=("Arial", 20, "bold"), relief=tk.RIDGE)
            btn.grid(row=1, column=2+i, padx=2, pady=(5, 0))
            self.player2_suit_buttons[suit] = btn
        
        # table cards section
        table_frame = ttk.LabelFrame(main_frame, text="Table Cards", padding="8")
        table_frame.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # table card variables
        self.table_ranks = []
        self.table_suits = []
        self.table_suit_buttons = []
        for i in range(5):
            rank_var = tk.StringVar(value=self.table_cards[i][0] if i < len(self.table_cards) and self.table_cards[i] else "")
            suit_var = tk.StringVar(value=self.table_cards[i][1] if i < len(self.table_cards) and self.table_cards[i] else "")
            self.table_ranks.append(rank_var)
            self.table_suits.append(suit_var)
            
            # Row layout: flop (0..2) on row 0, turn/river (3..4) on row 1
            row_idx = 0 if i < 3 else 1
            base_col = (i if i < 3 else (i - 3)) * 6
            
            ttk.Label(table_frame, text=f"{i+1}:").grid(row=row_idx, column=base_col, padx=(0, 2), pady=(0, 4))
            rank_entry = ttk.Entry(table_frame, textvariable=rank_var, width=3)
            rank_entry.grid(row=row_idx, column=base_col+1, padx=(0, 2), pady=(0, 4))
            self._bind_uppercase(rank_var)
            
            # Suit buttons for table cards
            btn_map = {}
            for j, (suit, emoji) in enumerate(self.suit_emojis.items()):
                fg = '#c0392b' if suit in ('h', 'd') else '#000000'
                btn = tk.Button(table_frame, text=emoji, width=2, height=1,
                               command=lambda s=suit, idx=i: self._set_table_suit(idx, s),
                               bg=self._suit_bg(suit_var.get(), suit),
                               fg=fg, font=("Arial", 20, "bold"), relief=tk.RIDGE)
                btn.grid(row=row_idx, column=base_col+2+j, padx=2, pady=(0, 4))
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
        
        # Start the GUI
        self.root.mainloop()
    
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

    def _load_calibration(self):
        """Load calibration file if present."""
        try:
            calib_path = 'screen_calibration.json'
            if os.path.exists(calib_path):
                with open(calib_path, 'r') as f:
                    data = json.load(f)
                if 'game_window' in data:
                    self.game_window = data['game_window']
                if 'screen_regions' in data:
                    self.screen_regions = data['screen_regions']
        except Exception:
            pass

    def _capture_screenshot(self):
        """Capture the current GOP3 game window as BGR numpy array."""
        try:
            from utils import GameWindowCapture, WindowDetector
            # Try to bring window forward softly
            try:
                wd = WindowDetector()
                wd.activate_game_window()
            except Exception:
                pass
            gwc = GameWindowCapture("GOP3")
            # Try locate window
            try:
                gwc.find_window()
            except Exception:
                pass
            img = None
            try:
                img = gwc.capture_game_image()
            except Exception:
                # fallback try direct capture
                try:
                    img = gwc.capture_window_image()
                except Exception:
                    img = None
            return img
        except Exception as e:
            try:
                messagebox.showwarning("Capture Failed", f"Could not capture game window: {e}")
            except Exception:
                pass
            return None

    def _detect_cards_from_screen(self):
        """Run detection and return (player_cards, table_cards)."""
        try:
            screenshot = self._capture_screenshot()
            if screenshot is None:
                return [], []
            player_cards = self.card_detector.get_player_cards(screenshot, self.screen_regions, self.game_window)
            table_cards = self.card_detector.get_table_cards(screenshot, self.screen_regions, self.game_window)
            return player_cards, table_cards
        except Exception as e:
            try:
                messagebox.showwarning("Detection Failed", f"Card detection failed: {e}")
            except Exception:
                pass
            return [], []

    def _fill_card_value(self, rank_var, suit_var, card_value):
        """Helper to set rank/suit variables from a 'Rs' string like 'Ah'."""
        if not card_value or len(card_value) < 2:
            return False
        rank = card_value[0].upper()
        suit = card_value[1].lower()
        if rank and suit in self.suits:
            rank_var.set(rank)
            suit_var.set(suit)
            return True
        return False

    def _update_state(self):
        """Detect current table and fill only blank fields with newly detected cards."""
        player_cards, table_cards = self._detect_cards_from_screen()
        # Update player cards if blank
        if (self.player_card1_rank.get() == '' or self.player_card1_suit.get() == '') and len(player_cards) >= 1 and player_cards[0]:
            self._fill_card_value(self.player_card1_rank, self.player_card1_suit, player_cards[0])
        if (self.player_card2_rank.get() == '' or self.player_card2_suit.get() == '') and len(player_cards) >= 2 and player_cards[1]:
            self._fill_card_value(self.player_card2_rank, self.player_card2_suit, player_cards[1])
        # Update table cards if blank
        for i in range(min(5, len(self.table_ranks))):
            if (self.table_ranks[i].get() == '' or self.table_suits[i].get() == '') and i < len(table_cards) and table_cards[i]:
                self._fill_card_value(self.table_ranks[i], self.table_suits[i], table_cards[i])
        # Refresh highlights
        self._update_suit_buttons()

    def _new_game(self):
        """Perform a fresh detection cycle akin to 'Play single hand' and refresh fields."""
        # Clear all to force new detection
        self.player_card1_rank.set('')
        self.player_card1_suit.set('')
        self.player_card2_rank.set('')
        self.player_card2_suit.set('')
        for i in range(5):
            self.table_ranks[i].set('')
            self.table_suits[i].set('')
        # Detect and then fill all available cards (player + table)
        self._update_state()
    
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

# Global instance
confirmation_window = CardConfirmationWindow()

def confirm_cards(detected_player_cards, detected_table_cards, extracted_images=None):
    """Show card confirmation window and return user's choice"""
    return confirmation_window.show_confirmation(detected_player_cards, detected_table_cards, extracted_images)
