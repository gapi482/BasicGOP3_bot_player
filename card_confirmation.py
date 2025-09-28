#!/usr/bin/env python3
"""
Fast Card Confirmation Window for Poker Bot
"""
import tkinter as tk
from tkinter import ttk, messagebox
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
        
        # Card options
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        self.suit_emojis = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
        self.suits = ['h', 'd', 'c', 's']
        self.card_options = [f"{rank}{suit}" for rank in self.ranks for suit in self.suits]
        
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
        
        # Load template images
        self.load_template_images()
        
        # Create window in a separate thread to avoid blocking
        self.result = None
        thread = threading.Thread(target=self._create_window)
        thread.daemon = True
        thread.start()
        
        # Wait for result
        while self.result is None:
            time.sleep(0.1)
        
        return self.result
    
    def _create_window(self):
        """Create the confirmation window"""
        self.root = tk.Tk()
        self.root.title("Card Confirmation")
        self.root.geometry("900x400")
        self.root.attributes('-topmost', True)  # Always on top
        self.root.focus_force()  # Force focus
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
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
        
        rank1_combo = ttk.Combobox(player_frame, textvariable=self.player_card1_rank, 
                                  values=self.ranks, width=3)
        rank1_combo.grid(row=0, column=1, padx=(0, 5))
        
        # Suit buttons for card 1
        for i, (suit, emoji) in enumerate(self.suit_emojis.items()):
            btn = tk.Button(player_frame, text=emoji, width=2, height=1,
                           command=lambda s=suit: self._set_suit(1, s),
                           bg='white' if self.player_card1_suit.get() == suit else 'lightgray')
            btn.grid(row=0, column=2+i, padx=1)
        
        # Player card 2
        ttk.Label(player_frame, text="Card 2:").grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        self.player_card2_rank = tk.StringVar(value=self.player_cards[1][0] if len(self.player_cards) > 1 and self.player_cards[1] else "")
        self.player_card2_suit = tk.StringVar(value=self.player_cards[1][1] if len(self.player_cards) > 1 and self.player_cards[1] else "")
        
        rank2_combo = ttk.Combobox(player_frame, textvariable=self.player_card2_rank, 
                                  values=self.ranks, width=3)
        rank2_combo.grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        
        # Suit buttons for card 2
        for i, (suit, emoji) in enumerate(self.suit_emojis.items()):
            btn = tk.Button(player_frame, text=emoji, width=2, height=1,
                           command=lambda s=suit: self._set_suit(2, s),
                           bg='white' if self.player_card2_suit.get() == suit else 'lightgray')
            btn.grid(row=1, column=2+i, padx=1, pady=(5, 0))
        
        # Community cards section
        community_frame = ttk.LabelFrame(main_frame, text="Community Cards", padding="8")
        community_frame.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Community card variables
        self.community_ranks = []
        self.community_suits = []
        for i in range(5):
            rank_var = tk.StringVar(value=self.community_cards[i][0] if i < len(self.community_cards) and self.community_cards[i] else "")
            suit_var = tk.StringVar(value=self.community_cards[i][1] if i < len(self.community_cards) and self.community_cards[i] else "")
            self.community_ranks.append(rank_var)
            self.community_suits.append(suit_var)
            
            ttk.Label(community_frame, text=f"{i+1}:").grid(row=0, column=i*6, padx=(0, 2))
            rank_combo = ttk.Combobox(community_frame, textvariable=rank_var, 
                                     values=self.ranks, width=3)
            rank_combo.grid(row=0, column=i*6+1, padx=(0, 2))
            
            # Suit buttons for community cards
            for j, (suit, emoji) in enumerate(self.suit_emojis.items()):
                btn = tk.Button(community_frame, text=emoji, width=2, height=1,
                               command=lambda s=suit, idx=i: self._set_community_suit(idx, s),
                               bg='white' if suit_var.get() == suit else 'lightgray')
                btn.grid(row=0, column=i*6+2+j, padx=1)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=6, pady=(15, 0))
        
        # Confirm button
        confirm_btn = ttk.Button(button_frame, text="CONFIRM & PLAY", 
                                command=self._confirm_cards)
        confirm_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Fold button
        fold_btn = ttk.Button(button_frame, text="FOLD", 
                             command=self._fold_hand)
        fold_btn.grid(row=0, column=1, padx=(0, 10))
        
        # Skip button
        skip_btn = ttk.Button(button_frame, text="SKIP", 
                             command=self._skip_confirmation)
        skip_btn.grid(row=0, column=2)
        
        # Bind Enter key to confirm
        self.root.bind('<Return>', lambda e: self._confirm_cards())
        self.root.bind('<Escape>', lambda e: self._fold_hand())
        
        # Focus on first combo box
        rank1_combo.focus()
        
        # Start the GUI
        self.root.mainloop()
    
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
        for i, (suit, emoji) in enumerate(self.suit_emojis.items()):
            btn = self.root.grid_slaves(row=0, column=2+i)[0]
            btn.config(bg='white' if self.player_card1_suit.get() == suit else 'lightgray')
        
        # Update player card 2 buttons
        for i, (suit, emoji) in enumerate(self.suit_emojis.items()):
            btn = self.root.grid_slaves(row=1, column=2+i)[0]
            btn.config(bg='white' if self.player_card2_suit.get() == suit else 'lightgray')
        
        # Update community card buttons
        for card_idx in range(5):
            for j, (suit, emoji) in enumerate(self.suit_emojis.items()):
                btn = self.root.grid_slaves(row=0, column=card_idx*6+2+j)[0]
                btn.config(bg='white' if self.community_suits[card_idx].get() == suit else 'lightgray')
    
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
