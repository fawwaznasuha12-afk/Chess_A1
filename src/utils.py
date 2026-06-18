import torch
import numpy as np
import chess
import os
import json
from datetime import datetime

class Utils:
    """Utility functions"""
    
    @staticmethod
    def save_model(model, path, metadata=None):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {}
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load_model(model, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {path}")
            return model, checkpoint.get('metadata', {})
        else:
            print(f"Model file {path} not found")
            return model, {}
    
    @staticmethod
    def board_to_fen(board):
        return board.fen()
    
    @staticmethod
    def fen_to_board(fen):
        return chess.Board(fen)
    
    @staticmethod
    def log_message(message, log_file="logs/training.log"):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
        print(message)
    
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    @staticmethod
    def create_training_data_file(positions, probs, values, filename="data/training_data.npz"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez_compressed(filename, positions=positions, probs=probs, values=values)
        print(f"Training data saved to {filename}")
    
    @staticmethod
    def load_training_data_file(filename="data/training_data.npz"):
        if os.path.exists(filename):
            data = np.load(filename)
            return data['positions'], data['probs'], data['values']
        else:
            print(f"Training data file {filename} not found")
            return None, None, None
