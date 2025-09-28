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
import threading
import time

class CardConfirmationWindow:
    def __init__(self):
        self.root = None
        self.confirmed_cards = None
        self.player_cards = []
        self.community_cards = []
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
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        self.suit_emojis = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
        self.suits = ['h', 'd', 'c', 's']
        self.card_options = [f"{rank}{suit}" for rank in self.ranks for suit in self.suits]
        
        # Button reference maps to avoid grid lookups
        self.player1_suit_buttons = {}
        self.player2_suit_buttons = {}
        self.community_suit_buttons = []  # list of dicts per community card

        # Calibration and capture/detector placeholders
        self.game_window = {
            'left': 0,
            'top': 40,
            'width': 1920,
            'height': 1000
        }
        self.screen_regions = {
            'player_card1': (899, 628, 60, 80),
            'player_card2': (973, 628, 60, 80),
            'flop_cards': [(540, 450, 70, 90), (620, 450, 70, 90), (700, 450, 70, 90)],
            'turn_card': (780, 450, 70, 90),
            'river_card': (860, 450, 70, 90)
        }
        
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
    
    def show_confirmation(self, detected_player_cards, detected_community_cards, extracted_images):
        """Show the card confirmation window"""
        self.player_cards = detected_player_cards.copy() if detected_player_cards else []
        self.community_cards = detected_community_cards.copy() if detected_community_cards else []
        self.extracted_images = extracted_images.copy() if extracted_images else {}
        
        # Create window on main thread (Tk must run in main thread)
        self.result = None
        self._create_window()
        return self.result
    
    def _create_window(self):
        """Create the confirmation window"""
        self.root = tk.Tk()
        self.root.title("Card Confirmation")
        self.root.geometry("980x460")
        self.root.attributes('-topmost', True)  # Always on top
        self.root.focus_force()  # Force focus
        
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
                           fg=fg, font=("Arial", 20, "bold"), relief=tk.RIDGE)
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
        
        # Community cards section
        community_frame = ttk.LabelFrame(main_frame, text="Community Cards", padding="8")
        community_frame.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Community card variables
        self.community_ranks = []
        self.community_suits = []
        self.community_suit_buttons = []
        for i in range(5):
            rank_var = tk.StringVar(value=self.community_cards[i][0] if i < len(self.community_cards) and self.community_cards[i] else "")
            suit_var = tk.StringVar(value=self.community_cards[i][1] if i < len(self.community_cards) and self.community_cards[i] else "")
            self.community_ranks.append(rank_var)
            self.community_suits.append(suit_var)
            
            # Row layout: flop (0..2) on row 0, turn/river (3..4) on row 1
            row_idx = 0 if i < 3 else 1
            base_col = (i if i < 3 else (i - 3)) * 6
            
            ttk.Label(community_frame, text=f"{i+1}:").grid(row=row_idx, column=base_col, padx=(0, 2), pady=(0, 4))
            rank_entry = ttk.Entry(community_frame, textvariable=rank_var, width=3)
            rank_entry.grid(row=row_idx, column=base_col+1, padx=(0, 2), pady=(0, 4))
            self._bind_uppercase(rank_var)
            
            # Suit buttons for community cards
            btn_map = {}
            for j, (suit, emoji) in enumerate(self.suit_emojis.items()):
                fg = '#c0392b' if suit in ('h', 'd') else '#000000'
                btn = tk.Button(community_frame, text=emoji, width=2, height=1,
                               command=lambda s=suit, idx=i: self._set_community_suit(idx, s),
                               bg=self._suit_bg(suit_var.get(), suit),
                               fg=fg, font=("Arial", 20, "bold"), relief=tk.RIDGE)
                btn.grid(row=row_idx, column=base_col+2+j, padx=2, pady=(0, 4))
                btn_map[suit] = btn
            self.community_suit_buttons.append(btn_map)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=6, pady=(15, 0))
        
        # Confirm button
        confirm_btn = ttk.Button(button_frame, text="CONFIRM & PLAY", 
                                command=self._confirm_cards, style='Accent.TButton')
        confirm_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Fold button
        fold_btn = ttk.Button(button_frame, text="FOLD", 
                             command=self._fold_hand)
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
    
    def _set_community_suit(self, card_idx, suit):
        """Set suit for community card"""
        self.community_suits[card_idx].set(suit)
        self._update_suit_buttons()
    
    def _update_suit_buttons(self):
        """Update suit button colors"""
        # Update player card 1 buttons
        for suit, btn in self.player1_suit_buttons.items():
            btn.configure(bg=self._suit_bg(self.player_card1_suit.get(), suit))
        
        # Update player card 2 buttons
        for suit, btn in self.player2_suit_buttons.items():
            btn.configure(bg=self._suit_bg(self.player_card2_suit.get(), suit))
        
        # Update community card buttons
        for idx, suit_map in enumerate(self.community_suit_buttons):
            for suit, btn in suit_map.items():
                btn.configure(bg=self._suit_bg(self.community_suits[idx].get(), suit))

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
        """Run detection and return (player_cards, community_cards)."""
        try:
            from card_detection import CardDetector
            screenshot = self._capture_screenshot()
            if screenshot is None:
                return [], []
            detector = CardDetector()
            player_cards = detector.get_player_cards(screenshot, self.screen_regions, self.game_window)
            community_cards = detector.get_community_cards(screenshot, self.screen_regions, self.game_window)
            return player_cards, community_cards
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
        player_cards, community_cards = self._detect_cards_from_screen()
        # Update player cards if blank
        if self.player_card1_rank.get() == '' and len(player_cards) >= 1 and player_cards[0]:
            self._fill_card_value(self.player_card1_rank, self.player_card1_suit, player_cards[0])
        if self.player_card2_rank.get() == '' and len(player_cards) >= 2 and player_cards[1]:
            self._fill_card_value(self.player_card2_rank, self.player_card2_suit, player_cards[1])
        # Update community cards if blank
        for i in range(min(5, len(self.community_ranks))):
            if self.community_ranks[i].get() == '' and i < len(community_cards) and community_cards[i]:
                self._fill_card_value(self.community_ranks[i], self.community_suits[i], community_cards[i])
        # Refresh highlights
        self._update_suit_buttons()

    def _new_game(self):
        """Reset all card fields; then refresh player/flop if present."""
        # Clear all
        self.player_card1_rank.set('')
        self.player_card1_suit.set('')
        self.player_card2_rank.set('')
        self.player_card2_suit.set('')
        for i in range(5):
            self.community_ranks[i].set('')
            self.community_suits[i].set('')
        # Detect current table and fill all detected positions for player and flop
        player_cards, community_cards = self._detect_cards_from_screen()
        if len(player_cards) >= 1 and player_cards[0]:
            self._fill_card_value(self.player_card1_rank, self.player_card1_suit, player_cards[0])
        if len(player_cards) >= 2 and player_cards[1]:
            self._fill_card_value(self.player_card2_rank, self.player_card2_suit, player_cards[1])
        # Fill flop (first three if present)
        for i in range(min(3, len(community_cards))):
            if community_cards[i]:
                self._fill_card_value(self.community_ranks[i], self.community_suits[i], community_cards[i])
        # Refresh highlights
        self._update_suit_buttons()
    
    def _confirm_cards(self):
        """Confirm the selected cards"""
        # Get player cards
        player_cards = []
        if self.player_card1_rank.get() and self.player_card1_suit.get():
            player_cards.append(f"{self.player_card1_rank.get()}{self.player_card1_suit.get()}")
        if self.player_card2_rank.get() and self.player_card2_suit.get():
            player_cards.append(f"{self.player_card2_rank.get()}{self.player_card2_suit.get()}")
        
        # Get community cards
        community_cards = []
        for i in range(5):
            if self.community_ranks[i].get() and self.community_suits[i].get():
                community_cards.append(f"{self.community_ranks[i].get()}{self.community_suits[i].get()}")
        
        self.result = {
            'action': 'confirm',
            'player_cards': player_cards,
            'community_cards': community_cards
        }
        
        self.root.destroy()
    
    def _fold_hand(self):
        """Fold the hand"""
        self.result = {
            'action': 'fold',
            'player_cards': [],
            'community_cards': []
        }
        
        self.root.destroy()
    
    def _skip_confirmation(self):
        """Skip confirmation and use detected cards"""
        self.result = {
            'action': 'skip',
            'player_cards': self.player_cards,
            'community_cards': self.community_cards
        }
        
        self.root.destroy()

# Global instance
confirmation_window = CardConfirmationWindow()

def confirm_cards(detected_player_cards, detected_community_cards, extracted_images=None):
    """Show card confirmation window and return user's choice"""
    return confirmation_window.show_confirmation(detected_player_cards, detected_community_cards, extracted_images)
